import os
from typing import List, Dict, Any

class DocumentPreprocessor:
    """
    대규모 텍스트 문서를 Mistral API 컨텍스트에 적합한 청크 단위로 분할합니다.
    """
    def __init__(self, max_tokens_per_chunk=1000, overlap=100):
        self.max_tokens = max_tokens_per_chunk
        self.overlap = overlap

    def chunk_text(self, text: str, source_name: str, start_id: int = 0) -> List[Dict[str, Any]]:
        """
        텍스트를 중첩(overlap)을 포함한 고정 크기 청크로 나눕니다.
        """
        # 대략적인 토큰-문자 변환 (1토큰 ~= 4자)
        char_limit = self.max_tokens * 4
        char_overlap = self.overlap * 4
        
        chunks = []
        start = 0
        current_id = start_id
        
        while start < len(text):
            end = start + char_limit
            content = text[start:end]
            
            chunks.append({
                "chunk_id": current_id,
                "source": source_name,
                "content": content,
                "length": len(content)
            })
            
            current_id += 1
            if end >= len(text):
                break
            start += (char_limit - char_overlap)
            
        return chunks