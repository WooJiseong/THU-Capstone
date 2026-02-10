import requests
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util

class MistralAPIClient:
    _instance = None

    def __init__(self, key_filename: str = "mistral_api.txt"):
        self._setup_key(key_filename)
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.model = "mistral-small-latest"
        
        # 1. 가벼우면서 성능이 좋은 SBERT 모델 로드 (최초 1회)
        # 'all-MiniLM-L6-v2'는 속도가 매우 빠르고 임베딩 성능이 준수합니다.
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. 기준점(Anchor) 설정: 사실 중심 vs 창의 중심
        self.factual_anchor = "Technical documentation analysis, extraction of facts, version numbers, dates, regulations, and objective evidence based on provided text."
        self.creative_anchor = "Creative writing, imaginative brainstorming, subjective suggestions, storytelling, and flexible conversation without strict constraints."
        
        # 기준점 임베딩 미리 계산
        self.factual_emb = self.st_model.encode(self.factual_anchor, convert_to_tensor=True)
        self.creative_emb = self.st_model.encode(self.creative_anchor, convert_to_tensor=True)

    def _setup_key(self, key_filename):
        # (기존 키 로드 로직...)
        current_file_path = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file_path)
        path = os.path.join(api_dir, key_filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.api_key = f.read().strip()
                return
        raise FileNotFoundError(f"API key file not found at {path}")

    def _calculate_dynamic_temperature(self, query: str) -> float:
        """
        SBERT 유사도를 기반으로 0.1 ~ 1.0 사이의 온도를 결정합니다.
        """
        query_emb = self.st_model.encode(query, convert_to_tensor=True)
        
        # 코사인 유사도 계산
        sim_to_factual = util.cos_sim(query_emb, self.factual_emb).item()
        sim_to_creative = util.cos_sim(query_emb, self.creative_emb).item()
        
        # 가중치 계산 (Softmax 스타일 또는 단순 비율)
        # 창의적 유사도가 높을수록 온도가 1.0에 가까워지도록 매핑
        # 분모가 0이 되는 것을 방지하기 위해 epsilon 추가
        sum_sim = sim_to_factual + sim_to_creative
        creative_weight = sim_to_creative / (sum_sim + 1e-9)
        
        # 선형 매핑: 0.1(Min) + 0.9 * creative_weight
        # 사실적인 질문일수록 weight가 낮아져 0.1에 수렴하게 됨
        temp = 0.1 + (0.9 * creative_weight)
        
        # 값의 범위를 0.1 ~ 1.0 사이로 클리핑
        return float(np.clip(temp, 0.1, 1.0))

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: int = 4096):
        last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
        
        # 온도가 명시되지 않았다면 SBERT로 계산
        if temperature is None:
            final_temp = self._calculate_dynamic_temperature(last_user_msg)
        else:
            final_temp = temperature

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": round(final_temp, 2), # 소수점 둘째자리까지
            "max_tokens": max_tokens
        }

        print(f"[AI Optimizer] Calculated Temperature: {round(final_temp, 2)}")

        for i in range(3):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                return str(response.json()["choices"][0]["message"]["content"])
            except Exception as e:
                time.sleep(2 ** i)
                return f"Error occurred: {str(e)}"

# --- 테스트 코드 ---
if __name__ == "__main__":
    client = MistralAPIClient()
    
    # 사실 중심 질문 테스트
    print("Test 1: Factual Query")
    client.chat_completion([{"role": "user", "content": "What is the specific build number for vSphere 7.0 Update 3?"}])
    
    # 창의 중심 질문 테스트
    print("\nTest 2: Creative Query")
    client.chat_completion([{"role": "user", "content": "Imagine a future where AI governs the world. How would the EU AI Act adapt?"}])