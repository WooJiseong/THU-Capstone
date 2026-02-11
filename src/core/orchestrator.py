import time
import os
import pickle
import yaml
import numpy as np
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from sentence_transformers import util

from src.api.client import MistralAPIClient
from src.data_loader.preprocessor import FastLongContextPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    Planner-Navigator-Executor 구조를 가진 에이전트형 오케스트레이터입니다.
    대규모 문서의 임베딩 캐싱 및 병렬 처리를 통해 성능을 최적화합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = MistralAPIClient()
        
        # 1. 프롬프트 YAML 로드
        self.prompts = self._load_yaml_prompts("configs/prompts.yaml")
        
        # 2. 전처리기 설정
        data_cfg = config.get('data', {})
        self.preprocessor = FastLongContextPreprocessor(
            max_tokens=data_cfg.get('chunk_size', 5000),
            overlap=data_cfg.get('overlap', 1000)
        )

        # 3. 임베딩 및 전략 설정
        self.batch_size = data_cfg.get('batch_size', 128)
        self.logger = ExperimentLogger()
        self.exp_name = config.get('experiment_name', 'agentic_long_context_v1')
        
        strat_cfg = config.get('strategy', {})
        self.landmark_top_n = strat_cfg.get('scan_top_n', 40)
        self.evidence_top_k = strat_cfg.get('final_top_k', 10)
        
        # 4. 캐시 경로 설정
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_yaml_prompts(self, path: str) -> Dict[str, Any]:
        """YAML 설정 파일에서 프롬프트 템플릿을 로드합니다."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"프롬프트 설정 파일을 찾을 수 없습니다: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_cache_path(self, file_name: str) -> str:
        """파일명 기반 안전한 캐시 경로 생성"""
        safe_name = "".join([c if c.isalnum() else "_" for c in file_name])
        return os.path.join(self.cache_dir, f"{safe_name}_v2.pkl")

    def _get_embeddings(self, chunks: List[Dict[str, Any]], file_name: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        임베딩 생성 가속 및 로컬 디스크 캐싱 로직.
        반환 타입: tuple(임베딩 배열, 청크 리스트)
        """
        cache_path = self._get_cache_path(file_name)
        
        # 1. 로컬 캐시 확인
        if os.path.exists(cache_path):
            print(f"[*] 로컬 캐시 발견: {cache_path} 로드 중...")
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    return cached_data["embeddings"], cached_data["chunks"]
            except Exception as e:
                print(f"[!] 캐시 로드 실패, 재생성합니다: {e}")

        # 2. 임베딩 생성 시작
        print(f"[*] 인베딩 생성 중: {len(chunks)} 청크 분석...")
        chunk_contents = [c['content'] for c in chunks]
        
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if device == "cpu":
            print(f"[*] CPU 멀티프로세스 가속 모드 가동 (Workers: {multiprocessing.cpu_count()-1})")
            pool = self.client.st_model.start_multi_process_pool()
            emb_np = self.client.st_model.encode_multi_process(
                chunk_contents, pool, batch_size=self.batch_size
            )
            self.client.st_model.stop_multi_process_pool(pool)
        else:
            print(f"[*] 가속 장치 가동: {device}")
            embeddings = self.client.st_model.encode(
                chunk_contents, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                batch_size=self.batch_size,
                device=device
            )
            emb_np = embeddings.cpu().numpy()

        # 3. 결과 캐싱
        with open(cache_path, "wb") as f:
            pickle.dump({"embeddings": emb_np, "chunks": chunks}, f)
        
        return emb_np, chunks

    def _create_search_plan(self, query: str) -> str:
        """Step 1: Planner - 질문 분석 및 탐색 전략 수립"""
        template = self.prompts['planner']['user']
        system_msg = self.prompts['planner']['system']
        
        return self.client.chat_completion([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": template.format(query=query)}
        ], temperature=0.0)

    def _scout_location(self, chunk: Dict[str, Any], plan: str, query: str) -> Dict[str, Any]:
        """Step 2: Navigator - 개별 청크의 가치 평가 및 증거 추출"""
        template = self.prompts['navigator']['user']
        system_msg = self.prompts['navigator']['system']
        
        content = chunk['content'][:3500]  # 컨텍스트 길이 최적화
        
        response = self.client.chat_completion([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": template.format(plan=plan, query=query, content=content)}
        ], temperature=0.0, max_tokens=300)
        
        score = 5
        if "VALUE_SCORE:" in response:
            try:
                score_part = response.split("VALUE_SCORE:")[1].split("\n")[0]
                score = int(''.join(filter(str.isdigit, score_part)))
            except: pass
            
        return {
            "score": score, 
            "findings": response, 
            "content": chunk['content'], 
            "metadata": chunk['metadata']
        }

    def _execute_synthesis(self, query: str, plan: str, top_findings: List[Dict[str, Any]]) -> str:
        """Step 3: Executor - 수집된 증거를 종합하여 최종 답변 작성"""
        template = self.prompts['synthesis']['user']
        system_msg = self.prompts['synthesis']['system']
        
        evidence_str = ""
        for i, f in enumerate(top_findings):
            loc = " > ".join(f['metadata'].get('hierarchy', ['Root']))
            evidence_str += f"<SOURCE index='{i+1}' location='{loc}'>\n{f['content']}\n</SOURCE>\n\n"

        return self.client.chat_completion([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": template.format(query=query, plan=plan, evidence=evidence_str)}
        ], max_tokens=2500)

    def run_pipeline(self, raw_text: str, query: str, file_name: str) -> str:
        """에이전트 파이프라인 통합 실행 메인 메서드"""
        start_time = time.time()
        
        # 1. 캐시 확인 및 전처리
        cache_path = self._get_cache_path(file_name)
        if os.path.exists(cache_path):
            chunks = []  # 임베딩 함수 내에서 캐시 로드됨
        else:
            chunks = self.preprocessor.preprocess(raw_text, {"file_name": file_name})
        
        # 2. 임베딩 가속 로드
        embeddings, chunks = self._get_embeddings(chunks, file_name)
        
        # 3. 에이전트 플래닝
        plan = self._create_search_plan(query)
        print(f"[*] 탐색 계획 수립 완료: {plan[:80]}...")
        
        # 4. 후보군 선별 (Cosine Similarity)
        query_emb = self.client.st_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
        
        top_indices = np.argsort(-scores)[:self.landmark_top_n]
        scout_indices = sorted(list(set(top_indices) | {0, len(chunks) - 1}))
        
        # 5. 에이전트 내비게이션 (Parallel Scouting)
        scout_results = []
        print(f"[*] {len(scout_indices)}개 위치에 대한 에이전트 스카우팅 개시...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_idx = {
                executor.submit(self._scout_location, chunks[idx], plan, query): idx 
                for idx in scout_indices
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(scout_indices), desc="Scouting"):
                scout_results.append(future.result())

        # 6. 증거 기반 최종 합성
        top_findings = sorted(scout_results, key=lambda x: x['score'], reverse=True)[:self.evidence_top_k]
        final_answer = self._execute_synthesis(query, plan, top_findings)
        
        self.logger.log_result(self.exp_name, query, final_answer, start_time, [file_name])
        return final_answer