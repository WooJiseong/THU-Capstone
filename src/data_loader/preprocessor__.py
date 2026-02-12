import re
import json
import multiprocessing
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class AdaptiveFastPreprocessor:
    """
    문서의 스타일(계층형 vs 단순 나열형)을 분석하여 최적의 청킹 전략을 실행합니다.
    """
    def __init__(
        self, 
        api_client: Any, 
        prompt_config: Dict[str, Any], 
        max_tokens: int = 3000, 
        overlap: int = 300
    ):
        self.client = api_client
        self.prompts = prompt_config
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap * 4
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """문서 샘플 분석을 통한 전략 수립 (JSON 모드 활용)"""
        samples = [text[:4000], text[len(text)//2:len(text)//2+4000], text[-4000:]]
        sample_text = "\n--- SAMPLE ---\n".join(samples)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self.prompts['style_analyzer']['system']},
                    {"role": "user", "content": self.prompts['style_analyzer']['user'].format(samples=sample_text)}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response)
        except Exception as e:
            print(f"[!] 스타일 분석 실패(기본값 사용): {e}")
            return {
                "strategy_type": "hierarchical",
                "primary_header_pattern": r'^(#{1,6})\s+(.+)$', 
                "chunk_size_multiplier": 1.0
            }

    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        스타일에 따라 계층적 분할 또는 고정 윈도우 분할을 수행합니다.
        """
        style_guide = self._analyze_style(raw_text)
        strategy = style_guide.get('strategy_type', 'hierarchical')
        
        sections = []
        
        # [개선] 전략에 따른 분기 처리
        if strategy == "fixed_window":
            # 헤더가 없는 경우 문서 전체를 하나의 섹션으로 간주하여 바로 청킹 단계로 전달
            sections = [{"title": "Whole Document", "content": raw_text, "hierarchy": ["Flat"]}]
            print(f"[*] 'fixed_window' 전략 적용: 헤더 없이 전체 문서 탐색")
        else:
            # 계층적(Hierarchical) 분할 시도
            pattern = style_guide.get('primary_header_pattern', r'^(#{1,6})\s+(.+)$')
            try:
                header_regex = re.compile(pattern, re.MULTILINE)
            except:
                header_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
                
            matches = list(header_regex.finditer(raw_text))
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

        # 2. 병렬 청킹
        all_chunks = []
        multiplier = style_guide.get('chunk_size_multiplier', 1.0)
        current_limit = int(self.max_chars * multiplier)
        current_overlap = int(self.overlap_chars * multiplier)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._chunk_task, s, metadata, current_limit, current_overlap) 
                for s in sections
            ]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(sections), desc="Adaptive Chunking"):
                all_chunks.extend(f.result())
        
        return all_chunks

    @staticmethod
    def _chunk_task(section: Dict[str, Any], meta: Dict[str, Any], limit: int, overlap: int) -> List[Dict[str, Any]]:
        content = section['content']
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + limit
            chunk_txt = content[start:end]
            
            if end < len(content):
                # 문장이나 줄바꿈 단위로 끊으려는 시도
                last_break = chunk_txt.rfind('\n')
                if last_break == -1: last_break = chunk_txt.rfind('. ')
                
                if last_break != -1 and last_break > (limit // 2):
                    actual_end = start + last_break + 1
                    chunk_txt = content[start:actual_end]
                else:
                    actual_end = end
            else:
                actual_end = end

            chunks.append({
                "content": f"[Context: {section['title']}]\n{chunk_txt.strip()}",
                "metadata": {**meta, "hierarchy": section['hierarchy'], "char_count": len(chunk_txt)}
            })
            
            start = actual_end - overlap
            if actual_end >= len(content) or start >= len(content): break
        return chunks