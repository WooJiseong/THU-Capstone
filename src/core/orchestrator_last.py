import time
import os
import pickle
import yaml
import numpy as np
import concurrent.futures
import regex as re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import torch
from sentence_transformers import util
from tqdm import tqdm

from src.data_loader.preprocessor import AdaptiveFastPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    RAG 파이프라인의 에이전트 워크플로우를 조율합니다.
    - 해싱 기반 캐시: 파일명 길이 제한 오류 방지
    - 3-Way Scouting: Planner, Navigator, Synthesis 전략
    - Refiner Layer: 사용자 경험을 위한 최종 응답 정제
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API 클라이언트 지연 임포트
        from src.api.client import MistralAPIClient
        self.client = MistralAPIClient()
        
        # 프롬프트 설정 로드
        self.prompts = self._load_yaml("configs/prompts.yaml")
        
        # 전처리기 초기화 (Single-level Fast Adaptive 방식)
        data_cfg = config.get('data', {})
        self.preprocessor = AdaptiveFastPreprocessor(
            self.client, 
            self.prompts, 
            max_tokens=data_cfg.get('chunk_size', 3000),
            overlap=data_cfg.get('overlap', 300)
        )
        
        # 로거 및 검색 파라미터 설정
        log_cfg = config.get('logging', {})
        self.logger = ExperimentLogger(base_path=log_cfg.get('base_path', 'results'))
        self.exp_name = config.get('experiment_name', 'v4_robust_pipeline')

        strat_cfg = config.get('strategy', {})
        self.landmark_top_n = strat_cfg.get('scan_top_n', 100)
        self.evidence_top_k = strat_cfg.get('final_top_k', 24)
        
        # 토큰 폭발 방지 가드레일 (상세 원문 포함 개수)
        self.max_raw_details = 6 
        
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _generate_cache_key(self, file_paths: List[str]) -> str:
        """[FIXED] 파일 경로 리스트를 해싱하여 안전한 캐시 키를 생성합니다."""
        combined_string = "|".join(sorted(file_paths))
        hash_val = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
        # 가독성을 위해 첫 파일명 일부 포함
        prefix = "".join([c if c.isalnum() else "_" for c in os.path.basename(file_paths[0])[:15]])
        return f"{prefix}_{hash_val}_v4_fast"

    def _get_embeddings(self, chunks: List[Dict[str, Any]], cache_key: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """임베딩을 로드하거나 새로 생성합니다."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data["embeddings"], data["chunks"]

        if not chunks:
            raise ValueError(f"캐시가 없고 전처리된 청크도 없습니다. Key: {cache_key}")

        print(f"[*] 임베딩 생성 중... ({len(chunks)} 청크)")
        texts = [c['content'] for c in chunks]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        embeddings = self.client.st_model.encode(
            texts, 
            batch_size=self.config.get('data', {}).get('batch_size', 256), 
            show_progress_bar=True, 
            device=device
        )
        
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "chunks": chunks}, f)
            
        return embeddings, chunks

    def _create_dynamic_plan(self, query: str) -> Tuple[int, str]:
        """Planner를 통해 탐색 계획 수립"""
        response = self.client.chat_completion([
            {"role": "system", "content": self.prompts['planner']['system']},
            {"role": "user", "content": self.prompts['planner']['user'].format(query=query)}
        ], temperature=0.0)
        
        top_n = self.landmark_top_n
        if "TOP_N:" in response:
            try:
                top_n = int(re.search(r'\d+', response.split("TOP_N:")[1]).group())
            except Exception:
                pass
        return top_n, response.strip()

    def run_pipeline(self, file_paths: List[str], query: str) -> str:
        """전체 RAG 워크플로우 실행"""
        start_time = time.time()
        cache_key = self._generate_cache_key(file_paths)
        
        try:
            # 1. 전처리 (경로 리스트 기반, 캐시 확인)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                all_chunks = []
                for path in file_paths:
                    file_chunks = self.preprocessor.preprocess(path, {"source": os.path.basename(path)})
                    all_chunks.extend(file_chunks)
                chunks = all_chunks
            else:
                chunks = [] 

            # 2. 임베딩 획득 및 플래닝
            embeddings, chunks = self._get_embeddings(chunks, cache_key)
            dynamic_k, plan = self._create_dynamic_plan(query)

            # 3. 벡터 유사도 검색
            query_emb = self.client.st_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, torch.from_numpy(embeddings))[0].cpu().numpy()
            top_indices = np.argsort(-scores)[:dynamic_k]
            
            # 서론/결론 보정
            terminals = set(range(0, min(5, len(chunks)))) | set(range(max(0, len(chunks)-5), len(chunks)))
            scout_indices = sorted(list(set(top_indices) | terminals))

            # 4. Navigator 스카우팅 (병렬 처리)
            scout_reports = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_idx = {}
                for idx in scout_indices:
                    chunk = chunks[idx]
                    pos_info = f"{(idx/len(chunks))*100:.1f}% 지점 ({chunk['metadata'].get('source')})"
                    
                    # Parent-Child 방식과 호환되도록 get() 사용
                    context_to_send = chunk.get('parent_content', chunk['content'])
                    
                    f = executor.submit(self.client.chat_completion, [
                        {"role": "system", "content": self.prompts['navigator']['system']},
                        {"role": "user", "content": self.prompts['navigator']['user'].format(
                            plan=plan, query=query, content=context_to_send[:4000],
                            position_info=pos_info, chunk_index=idx, total_chunks=len(chunks)
                        )}
                    ], temperature=0.0)
                    future_to_idx[f] = idx
                
                for f in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(scout_indices), desc="Scouting"):
                    report_text = f.result()
                    idx = future_to_idx[f]
                    
                    score = 5
                    if "VALUE_SCORE:" in report_text:
                        try: score = int(re.search(r'\d+', report_text.split("VALUE_SCORE:")[1]).group())
                        except: pass
                    
                    scout_reports.append({
                        "score": score,
                        "findings": report_text,
                        "raw_content": chunks[idx].get('parent_content', chunks[idx]['content']),
                        "source": chunks[idx]['metadata'].get('source')
                    })

            # 5. Evidence Packing
            sorted_reports = sorted(scout_reports, key=lambda x: x['score'], reverse=True)[:self.evidence_top_k]
            packed_evidence = ""
            for i, rep in enumerate(sorted_reports):
                packed_evidence += f"\n[Evidence {i+1} from {rep['source']}]\n{rep['findings']}\n"
                if i < self.max_raw_details:
                    packed_evidence += f"Detailed Excerpt: {rep['raw_content'][:2000]}\n"

            # 6. 'Auditor' 기술 보고서 합성
            print(f"[*] 단계 6: 상세 기술 보고서 합성 중...")
            technical_report = self.client.chat_completion([
                {"role": "system", "content": self.prompts['synthesis']['system']},
                {"role": "user", "content": self.prompts['synthesis']['user'].format(
                    query=query, evidence=packed_evidence
                )}
            ])

            # 7. 'Communication Expert' 응답 정제
            print(f"[*] 단계 7: 사용자용 최종 응답 정제 중...")
            final_answer = self.client.chat_completion([
                {"role": "system", "content": self.prompts['refiner']['system']},
                {"role": "user", "content": self.prompts['refiner']['user'].format(
                    query=query, report=technical_report
                )}
            ])

            # 8. 로깅 및 반환
            self.logger.log_result(self.exp_name, query, final_answer, start_time, file_paths)
            return final_answer

        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            self.logger.log_result(self.exp_name, query, error_msg, start_time, file_paths)
            print(f"[!] 파이프라인 에러: {e}")
            return error_msg