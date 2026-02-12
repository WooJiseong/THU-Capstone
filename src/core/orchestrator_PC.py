import time
import os
import pickle
import yaml
import numpy as np
import concurrent.futures
import regex as re
import hashlib
from typing import List, Dict, Any, Tuple
import torch
from sentence_transformers import util
from tqdm import tqdm

from src.data_loader.preprocessor import AdaptiveFastPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    여러 파일 경로를 받아 해싱 기반 캐싱과 Parent-Child RAG를 조율합니다.
    Refiner 레이어를 포함하여 최종 답변 품질을 최적화합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        from src.api.client import MistralAPIClient
        self.client = MistralAPIClient()
        self.prompts = self._load_yaml("configs/prompts.yaml")
        
        data_cfg = config.get('data', {})
        self.preprocessor = AdaptiveFastPreprocessor(
            self.client, self.prompts, 
            parent_size=data_cfg.get('chunk_size', 2500), 
            child_size=500
        )
        
        log_cfg = config.get('logging', {})
        self.logger = ExperimentLogger(base_path=log_cfg.get('base_path', 'results'))
        self.exp_name = config.get('experiment_name', 'v4_parent_child_hash')

        strat_cfg = config.get('strategy', {})
        self.landmark_top_n = strat_cfg.get('scan_top_n', 100)
        self.evidence_top_k = strat_cfg.get('final_top_k', 24)
        self.max_raw_details = 6 
        
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _generate_cache_key(self, file_paths: List[str]) -> str:
        """[FIXED] 파일 경로를 해싱하여 고유하고 안전한 파일명을 생성합니다."""
        combined = "|".join(sorted(file_paths))
        hash_val = hashlib.md5(combined.encode('utf-8')).hexdigest()
        prefix = os.path.basename(file_paths[0])[:15]
        return f"{prefix}_{hash_val}_v4"

    def _get_embeddings(self, chunks: List[Dict], cache_key: str):
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data["embeddings"], data["chunks"]

        print(f"[*] 통합 임베딩 생성 중... ({len(chunks)} 자식 청크)")
        texts = [c['content'] for c in chunks]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = self.client.st_model.encode(texts, batch_size=256, show_progress_bar=True, device=device)
        
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "chunks": chunks}, f)
        return embeddings, chunks

    def run_pipeline(self, file_paths: List[str], query: str) -> str:
        start_time = time.time()
        cache_key = self._generate_cache_key(file_paths)
        
        try:
            # 1. 전처리 (경로 리스트 기반)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if not os.path.exists(cache_path):
                all_chunks = []
                for path in file_paths:
                    file_chunks = self.preprocessor.preprocess(path, {"source": os.path.basename(path)})
                    all_chunks.extend(file_chunks)
                chunks = all_chunks
            else:
                chunks = [] 

            # 2. 임베딩 및 플래닝
            embeddings, chunks = self._get_embeddings(chunks, cache_key)
            dynamic_k, plan = self._create_dynamic_plan(query)

            # 3. 벡터 검색 (자식 기준)
            query_emb = self.client.st_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, torch.from_numpy(embeddings))[0].cpu().numpy()
            top_indices = np.argsort(-scores)[:dynamic_k]
            terminals = set(range(0, min(5, len(chunks)))) | set(range(max(0, len(chunks)-5), len(chunks)))
            scout_indices = sorted(list(set(top_indices) | terminals))

            # 4. 에이전트 스카우팅 (Navigator - 부모 맥락 분석)
            scout_reports = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_idx = {}
                for idx in scout_indices:
                    chunk = chunks[idx]
                    context = chunk.get('parent_content', chunk['content'])
                    f = executor.submit(self.client.chat_completion, [
                        {"role": "system", "content": self.prompts['navigator']['system']},
                        {"role": "user", "content": self.prompts['navigator']['user'].format(
                            plan=plan, query=query, content=context[:4500],
                            position_info=f"{idx/len(chunks)*100:.1f}%", chunk_index=idx, total_chunks=len(chunks)
                        )}
                    ], temperature=0.0)
                    future_to_idx[f] = idx
                
                for f in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(scout_indices), desc="Scouting"):
                    report = f.result()
                    score = 5
                    if "VALUE_SCORE:" in report:
                        try: score = int(re.search(r'\d+', report.split("VALUE_SCORE:")[1]).group())
                        except: pass
                    scout_reports.append({
                        "score": score, "findings": report,
                        "raw_content": chunks[future_to_idx[f]].get('parent_content', ""),
                        "source": chunks[future_to_idx[f]]['metadata'].get('source')
                    })

            # 5. Packing & Synthesis & Refine
            sorted_reports = sorted(scout_reports, key=lambda x: x['score'], reverse=True)[:self.evidence_top_k]
            evidence_text = ""
            for i, rep in enumerate(sorted_reports):
                evidence_text += f"\n[Evidence {i+1} from {rep['source']}]\n{rep['findings']}\n"
                if i < self.max_raw_details:
                    evidence_text += f"Detail: {rep['raw_content'][:1500]}\n"

            print(f"[*] 'Auditor' 합성 및 'Refiner' 정제 중...")
            report = self.client.chat_completion([
                {"role": "system", "content": self.prompts['synthesis']['system']},
                {"role": "user", "content": self.prompts['synthesis']['user'].format(query=query, evidence=evidence_text)}
            ])
            final_answer = self.client.chat_completion([
                {"role": "system", "content": self.prompts['refiner']['system']},
                {"role": "user", "content": self.prompts['refiner']['user'].format(query=query, report=report)}
            ])

            self.logger.log_result(self.exp_name, query, final_answer, start_time, file_paths)
            return final_answer
        except Exception as e:
            return f"Error: {str(e)}"

    def _create_dynamic_plan(self, query: str) -> Tuple[int, str]:
        response = self.client.chat_completion([
            {"role": "system", "content": self.prompts['planner']['system']},
            {"role": "user", "content": self.prompts['planner']['user'].format(query=query)}
        ], temperature=0.0)
        top_n = self.landmark_top_n
        if "TOP_N:" in response:
            try: top_n = int(re.search(r'\d+', response.split("TOP_N:")[1]).group())
            except: pass
        return top_n, response.strip()