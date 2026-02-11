import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

class GeneralMarkdownPreprocessor(BasePreprocessor):
    """
    고성능 노이즈 필터링 및 토큰 최적화 전처리기입니다.
    의미 없는 공백을 제거하고 구조를 보존하며 청킹합니다.
    """
    def __init__(self, max_tokens: int = 6000, overlap: int = 1000):
        # 1토큰 ~= 4자 기준
        self.max_chars = max_tokens * 4
        self.overlap = overlap
        self.header_pattern = re.compile(r'^#+\s')

    def _clean_content(self, text: str) -> str:
        """
        토큰 절약을 위한 텍스트 정제 로직:
        1. 연속된 줄바꿈을 2개로 압축
        2. 라인 끝의 불필요한 공백 제거
        3. 너무 짧거나 의미 없는 라인(예: 단독 특수문자) 필터링
        """
        # 줄바꿈 압축
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # 너무 짧은 노이즈 라인 필터링 (단, 헤더나 구분자는 유지)
            if len(stripped) < 2 and stripped not in ['#', '-', '*', '|']:
                continue
            cleaned_lines.append(stripped)
            
        return '\n'.join(cleaned_lines)

    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        cleaned_text = self._clean_content(raw_text)
        paragraphs = cleaned_text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for p in paragraphs:
            p_len = len(p) + 2
            
            # 단일 단락이 너무 큰 경우 (강제 분할)
            if p_len > self.max_chars:
                if current_chunk:
                    chunks.append(self._create_chunk('\n\n'.join(current_chunk), metadata))
                    current_chunk = []
                    current_length = 0
                
                # 거대 단락 분할
                start = 0
                while start < p_len:
                    end = start + self.max_chars
                    chunks.append(self._create_chunk(p[start:end], metadata))
                    start += (self.max_chars - self.overlap)
                continue

            if current_length + p_len > self.max_chars:
                chunks.append(self._create_chunk('\n\n'.join(current_chunk), metadata))
                # Overlap: 이전 청크의 마지막 요소 일부 유지
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_length = sum(len(x) + 2 for x in current_chunk)

            current_chunk.append(p)
            current_length += p_len
            
        if current_chunk:
            chunks.append(self._create_chunk('\n\n'.join(current_chunk), metadata))
            
        return chunks

    def _create_chunk(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": content,
            "metadata": {
                **metadata,
                "breadcrumbs": [line.strip('# ') for line in content.split('\n') if self.header_pattern.match(line)][-3:],
                "char_count": len(content)
            }
        }