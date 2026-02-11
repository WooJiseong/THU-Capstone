import requests
import time
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util

class MistralAPIClient:
    """
    적응형 Temperature 모델을 갖춘 Mistral API 클라이언트입니다.
    """
    def __init__(self, key_filename: str = "mistral_api.txt"):
        self._setup_key(key_filename)
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"
        self.model = "mistral-small-2506"
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.factual_anchor = "Technical documentation analysis, extraction of facts, version numbers, dates, regulations, and objective evidence."
        self.creative_anchor = "Creative writing, imaginative brainstorming, subjective suggestions, and storytelling."
        
        self.factual_emb = self.st_model.encode(self.factual_anchor, convert_to_tensor=True)
        self.creative_emb = self.st_model.encode(self.creative_anchor, convert_to_tensor=True)

    def _setup_key(self, key_filename: str):
        current_file_path = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file_path)
        path = os.path.join(api_dir, key_filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.api_key = f.read().strip()
                return
        raise FileNotFoundError(f"API key file not found at {path}")

    def _calculate_dynamic_temperature(self, query: str) -> float:
        query_emb = self.st_model.encode(query, convert_to_tensor=True)
        sim_to_factual = float(util.cos_sim(query_emb, self.factual_emb).item())
        sim_to_creative = float(util.cos_sim(query_emb, self.creative_emb).item())
        
        sum_sim = sim_to_factual + sim_to_creative
        creative_weight = sim_to_creative / (sum_sim + 1e-9)
        temp = 0.1 + (0.9 * creative_weight)
        return float(np.clip(temp, 0.1, 1.0))

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Mistral API 호출을 수행하며 항상 문자열(str) 반환을 보장합니다.
        """
        last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), "")
        final_temp = temperature if temperature is not None else self._calculate_dynamic_temperature(last_user_msg)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": round(float(final_temp), 2),
            "max_tokens": max_tokens
        }

        for i in range(3):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return str(content) if content is not None else "Error: Received empty content from API."
            except Exception as e:
                if i == 2:
                    return f"Error: API call failed after retries. {str(e)}"
                time.sleep(2 ** i)
        
        return "Error: Unexpected termination of API client loop."