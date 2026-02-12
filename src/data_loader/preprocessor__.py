import re
import json
import multiprocessing
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm

class AdaptiveFastPreprocessor:
    def __init__(self, api_client, prompt_config: Dict[str, Any], max_tokens: int = 3000, overlap: int = 500):
        self.client = api_client
        self.prompts = prompt_config
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap * 4
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """문서 샘플 분석을 통한 전략 수립"""
        samples = [text[:4000], text[len(text)//2:len(text)//2+4000], text[-4000:]]
        sample_text = "\n--- SAMPLE ---\n".join(samples)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.prompts['style_analyzer']['system']},
                    {"role": "user", "content": self.prompts['style_analyzer']['user'].format(samples=sample_text)}
                ],
                response_format={"type": "json_object"} # [핵심] JSON 모드 명시
            )
            return json.loads(response)
        except Exception as e:
            print(f"[!] 스타일 분석 실패(기본값 사용): {e}")
            return {"primary_header_pattern": r'^(#{1,6})\s+(.+)$', "chunk_size_multiplier": 1.0}

    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        style_guide = self._analyze_style(raw_text)
        pattern = style_guide.get('primary_header_pattern', r'^(#{1,6})\s+(.+)$')
        
        # 1. 섹션 분할 (정규표현식 기반)
        try:
            header_regex = re.compile(pattern, re.MULTILINE)
        except:
            header_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            
        matches = list(header_regex.finditer(raw_text))
        sections = []
        if not matches:
            sections = [{"title": "Full Document", "content": raw_text, "hierarchy": ["Root"]}]
        else:
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(raw_text)
                sections.append({
                    "title": m.group(0)[:100].strip(),
                    "content": raw_text[start:end],
                    "hierarchy": [m.group(0).strip()]
                })

        # 2. 병렬 청킹 (가드레일: 섹션이 너무 크면 강제 분할)
        all_chunks = []
        # 토큰 설정값 반영
        current_limit = int(self.max_chars * style_guide.get('chunk_size_multiplier', 1.0))
        current_overlap = int(self.overlap_chars * style_guide.get('chunk_size_multiplier', 1.0))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._chunk_task, s, metadata, current_limit, current_overlap) for s in sections]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(sections), desc="Adaptive Chunking"):
                all_chunks.extend(f.result())
        
        return all_chunks

    @staticmethod
    def _chunk_task(section: Dict[str, Any], meta: Dict[str, Any], limit: int, overlap: int) -> List[Dict[str, Any]]:
        """물리적 한계를 넘지 않도록 보장하는 청킹 워커"""
        content = section['content']
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + limit
            chunk_txt = content[start:end]
            
            # 문장 경계 보존 시도
            if end < len(content):
                last_period = chunk_txt.rfind('. ')
                if last_period != -1 and last_period > (limit // 2):
                    actual_end = start + last_period + 1
                    chunk_txt = content[start:actual_end]
                else:
                    actual_end = end
            else:
                actual_end = end

            chunks.append({
                "content": f"[Location: {section['title']}]\n{chunk_txt.strip()}",
                "metadata": {**meta, "hierarchy": section['hierarchy'], "char_count": len(chunk_txt)}
            })
            
            start = actual_end - overlap
            if actual_end >= len(content) or start >= len(content): break
        return chunks