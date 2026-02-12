import os
import re
import json
import multiprocessing
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Docling: 고급 PDF 레이아웃 분석 (무거움)
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None

class AdaptiveFastPreprocessor:
    """
    단일 계층(Single-level) 기반의 빠른 전처리기입니다.
    PDF 처리 시 Docling 실패에 대비한 강력한 다중 폴백(Fallback) 엔진을 포함합니다.
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
        
        # Docling 컨버터 초기화
        if DocumentConverter:
            self.converter = DocumentConverter()
        else:
            self.converter = None

    def preprocess(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """파일 확장자에 따라 최적의 전처리 엔진 선택"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._preprocess_pdf(file_path, metadata)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            return self._run_adaptive_chunking(raw_text, metadata)

    def _preprocess_pdf(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        [고도화된 PDF 추출 로직]
        1순위: Docling (레이아웃 보존)
        2순위: PyMuPDF (빠르고 정확한 텍스트 추출)
        3순위: PyPDF2 (최후의 기본 추출기)
        """
        text = ""
        print(f"[*] PDF 텍스트 추출 시작: {os.path.basename(file_path)}")
        
        # 1. Docling 시도
        if self.converter:
            try:
                conversion_result = self.converter.convert(file_path)
                text = conversion_result.document.export_to_markdown()
            except Exception as e:
                print(f"  [!] Docling 변환 실패 ({e}). 다른 엔진으로 전환합니다.")

        # 2. 폴백 엔진 (PyMuPDF / PyPDF2) 가동
        if not text.strip():
            print("  [*] 기본 PDF 추출 엔진(PyMuPDF/PyPDF2)을 가동합니다...")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = "\n".join([page.get_text() for page in doc])
            except ImportError:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                except ImportError:
                    print("  [Error] PDF 처리를 위한 라이브러리가 없습니다. 'pip install pymupdf'를 실행하세요.")
                    return []
            except Exception as fallback_e:
                print(f"  [Error] 폴백 엔진 변환 실패: {fallback_e}")
                return []

        if not text.strip():
            print(f"  [Error] {file_path}에서 텍스트를 추출하지 못했습니다. (스캔된 이미지일 수 있음)")
            return []
            
        print("  [+] PDF 텍스트 추출 완료. 청킹(Chunking)을 시작합니다.")
        return self._run_adaptive_chunking(text, metadata)

    def _run_adaptive_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        style_guide = self._analyze_style(text)
        strategy = style_guide.get('strategy_type', 'hierarchical')
        
        sections = []
        if strategy == "fixed_window":
            sections = [{"title": "Whole Document", "content": text, "hierarchy": ["Flat"]}]
        else:
            pattern = style_guide.get('primary_header_pattern', r'^(#{1,6})\s+(.+)$')
            try:
                header_regex = re.compile(pattern, re.MULTILINE)
            except:
                header_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            
            matches = list(header_regex.finditer(text))
            if not matches:
                sections = [{"title": "Full Document", "content": text, "hierarchy": ["Root"]}]
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
        multiplier = style_guide.get('chunk_size_multiplier', 1.0)
        limit = int(self.max_chars * multiplier)
        overlap = int(self.overlap_chars * multiplier)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._chunk_task, s, metadata, limit, overlap) for s in sections]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(sections), desc="Chunking"):
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
                "metadata": {**meta, "hierarchy": section['hierarchy']}
            })
            start = actual_end - overlap
            if actual_end >= len(content) or start >= len(content): break
        return chunks

    def _analyze_style(self, text: str) -> Dict[str, Any]:
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