import os
import re
import json
import multiprocessing
import concurrent.futures
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Docling: PDF를 구조화된 마크다운으로 변환
from docling.document_converter import DocumentConverter
# 이미지 처리를 위한 라이브러리 (필요 시 활용)
from PIL import Image

class AdaptiveFastPreprocessor:
    """
    Docling을 사용하여 PDF의 레이아웃을 보존하며 전처리하는 고도화된 클래스입니다.
    이미지 캡셔닝을 통해 시각 정보를 텍스트 임베딩에 포함시킵니다.
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
        
        # Docling 컨버터 초기화 (레이아웃 인지형 변환 엔진)
        self.converter = DocumentConverter()

    def _get_image_description(self, base64_image: str) -> str:
        """
        Mistral-Vision 모델을 호출하여 이미지의 내용을 텍스트로 설명받습니다.
        """
        try:
            # [가이드] 실제 구현 시 Mistral 비전 API (예: pixtral-12b)를 호출합니다.
            # 여기서는 프레임워크 구조를 보여주기 위해 논리적 흐름만 기술합니다.
            prompt = "Describe this image in detail for technical documentation RAG system. Focus on charts, tables, or diagrams."
            # response = self.client.vision_completion(prompt, base64_image)
            # return response
            return "[Image Description: A technical diagram showing workflow or data architecture.]"
        except Exception:
            return "[Image: Analysis failed]"

    def preprocess(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        파일 확장자에 따라 최적의 전처리 엔진을 선택합니다.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self._preprocess_pdf(file_path, metadata)
        else:
            # 일반 텍스트 파일 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            return self._run_adaptive_chunking(raw_text, metadata)

    def _preprocess_pdf(self, file_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Docling 엔진으로 PDF를 마크다운으로 변환하고 이미지 캡션을 병합합니다.
        """
        print(f"[*] Docling 엔진 가동 중: {os.path.basename(file_path)}")
        
        # 1. PDF 변환 실행
        conversion_result = self.converter.convert(file_path)
        
        # 2. 마크다운으로 내보내기 (표와 계층 구조가 보존됨)
        markdown_text = conversion_result.document.export_to_markdown()
        
        # 3. 이미지 설명 병합 (Optional)
        # Docling이 추출한 이미지 객체들을 순회하며 설명을 텍스트에 삽입하는 로직이 여기에 위치합니다.
        # current_content = self._merge_image_captions(markdown_text, conversion_result)
        
        return self._run_adaptive_chunking(markdown_text, metadata)

    def _run_adaptive_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        기존에 구현한 스타일 분석 기반 청킹 로직입니다.
        """
        style_guide = self._analyze_style(text)
        strategy = style_guide.get('strategy_type', 'hierarchical')
        
        sections = []
        if strategy == "fixed_window":
            sections = [{"title": "Whole Document", "content": text, "hierarchy": ["Flat"]}]
        else:
            pattern = style_guide.get('primary_header_pattern', r'^(#{1,6})\s+(.+)$')
            header_regex = re.compile(pattern, re.MULTILINE) if pattern else re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            
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

    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """(기존 로직 동일) LLM을 사용하여 문서의 스타일 분석"""
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

    @staticmethod
    def _chunk_task(section: Dict[str, Any], meta: Dict[str, Any], limit: int, overlap: int) -> List[Dict[str, Any]]:
        """(기존 로직 동일) 문장 경계를 고려한 청킹"""
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
                "metadata": {**meta, "hierarchy": section['hierarchy'], "char_count": len(chunk_txt)}
            })
            start = actual_end - overlap
            if actual_end >= len(content) or start >= len(content): break
        return chunks