import os
import re
import json
import multiprocessing
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Docling: PDF 레이아웃 분석 및 마크다운 변환
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

class AdaptiveFastPreprocessor:
    """
    스타일 분석과 Parent-Child Chunking을 수행하는 전처리기입니다.
    파일 경로를 받아 PDF/TXT 여부를 판단하고 계층 구조를 생성합니다.
    """
    def __init__(
        self, 
        api_client: Any, 
        prompt_config: Dict[str, Any], 
        parent_size: int = 2500, 
        child_size: int = 500,
        overlap: int = 200
    ):
        self.client = api_client
        self.prompts = prompt_config
        self.parent_chars = parent_size * 4
        self.child_chars = child_size * 4
        self.overlap_chars = overlap * 4
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        if DocumentConverter:
            self.converter = DocumentConverter()
        else:
            self.converter = None

    def preprocess(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        파일 형식별로 분기하여 최종적으로 Parent-Child 청크 리스트를 생성합니다.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf' and self.converter:
            return self._preprocess_pdf(file_path, metadata)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            return self._run_adaptive_hierarchical_chunking(raw_text, metadata)

    def _preprocess_pdf(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Docling을 사용하여 PDF 레이아웃 보존 마크다운 생성"""
        print(f"[*] PDF 레이아웃 분석 중: {os.path.basename(file_path)}")
        try:
            result = self.converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            return self._run_adaptive_hierarchical_chunking(markdown_text, metadata)
        except Exception as e:
            print(f"[!] PDF 변환 실패: {e}")
            return []

    def _run_adaptive_hierarchical_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """스타일 분석 후 섹션별로 Parent-Child 청킹 수행"""
        # 스타일 분석 호출
        style_guide = self._analyze_style(text)
        strategy = style_guide.get('strategy_type', 'hierarchical')
        
        sections = []
        if strategy == "fixed_window":
            sections = [{"title": "Main", "content": text, "hierarchy": ["Flat"]}]
        else:
            pattern = style_guide.get('primary_header_pattern', r'^(#{1,6})\s+(.+)$')
            try:
                header_regex = re.compile(pattern, re.MULTILINE)
            except:
                header_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            
            matches = list(header_regex.finditer(text))
            if not matches:
                sections = [{"title": "Document", "content": text, "hierarchy": ["Root"]}]
            else:
                for i, m in enumerate(matches):
                    start = m.start()
                    end = matches[i+1].start() if i+1 < len(matches) else len(text)
                    sections.append({
                        "title": m.group(0)[:100].strip(),
                        "content": text[start:end],
                        "hierarchy": [m.group(0).strip()]
                    })

        all_chunks = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._create_parent_child_worker, 
                    s, metadata, self.parent_chars, self.child_chars, self.overlap_chars
                ) for s in sections
            ]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(sections), desc="Hierarchical Chunking"):
                all_chunks.extend(f.result())
        return all_chunks

    @staticmethod
    def _create_parent_child_worker(section, meta, p_limit, c_limit, overlap) -> List[Dict]:
        content = section['content']
        chunks = []
        p_start = 0
        while p_start < len(content):
            p_end = min(p_start + p_limit, len(content))
            parent_txt = content[p_start:p_end].strip()
            
            c_start = 0
            while c_start < len(parent_txt):
                c_end = min(c_start + c_limit, len(parent_txt))
                child_txt = parent_txt[c_start:c_end].strip()
                
                chunks.append({
                    "content": child_txt,
                    "parent_content": f"[Source: {section['title']}]\n{parent_txt}",
                    "metadata": {**meta, "hierarchy": section['hierarchy']}
                })
                c_start += (c_limit - overlap)
                if c_end >= len(parent_txt): break
            p_start += (p_limit - overlap)
            if p_end >= len(content): break
        return chunks

    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """(기존 동일) 문서 스타일 분석"""
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
        except:
            return {"strategy_type": "hierarchical", "chunk_size_multiplier": 1.0}