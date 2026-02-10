import os
from typing import Optional, Any
from src.api.client import MistralAPIClient

class ModelManager:
    """
    Mistral API를 사용하여 기술 문서 분석을 수행하는 매니저입니다.
    SBERT 기반 동적 온도 조절 기능을 클라이언트와 연동합니다.
    """

    def __init__(self, model_id: str = "mistral-small-latest"):
        """
        Args:
            model_id: 사용할 모델 ID (최신 snapshot 권장)
        """
        self.model_id = model_id
        # 클라이언트 초기화 시 SBERT 모델이 함께 로드됩니다.
        self.api_client = MistralAPIClient()

    def load_model(self):
        """
        API 연결 상태와 페르소나 설정을 확인합니다.
        """
        print(f"--- [API Mode Active: {self.model_id}] ---")
        print("Model Persona: Professional Technical Documentation Analyst")
        print("Adaptive Parameter Tuning: SBERT-based Semantic Analysis Enabled")

    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: Optional[float] = None) -> str:
        """
        SBERT 기반 동적 온도를 활용하여 답변 정밀도를 조절합니다.
        """
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an expert Technical Documentation Analyst. "
                    "Analyze the provided context meticulously. Answer based strictly on facts: "
                    "dates, version numbers, and specific technical requirements. "
                    "If the information is missing, state that you don't know."
                )
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # chat_completion의 결과가 None일 가능성에 대비해 빈 문자열이나 에러 메시지를 기본값으로 설정
        result = self.api_client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens
        )

        # 분석기(Pylance/Mypy)에게 이 값이 확실히 str임을 보장해줍니다.
        if result is None:
            return "Error: Failed to retrieve a response from the API."
            
        return str(result)