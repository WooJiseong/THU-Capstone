import time
import os
import pickle
import yaml
import numpy as np
import concurrent.futures
import regex as re
from typing import List, Dict, Any, Tuple, Optional
import torch
from sentence_transformers import util
from tqdm import tqdm

from src.data_loader.preprocessor import AdaptiveFastPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    전체 LLM 파이프라인의 워크플로우를 조율하는 핵심 모듈입니다.
    Planner - Navigator - Executor(Synthesis)의 3단계 전략을 수행하며,
    대규모 문서 처리를 위한 토큰 최적화 로직이 포함되어 있습니다.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 실험 설정 (data, strategy, logging 관련 파라미터 포함)
        """
        self.config = config
        
        # 순환 참조 방지를 위해 내부 임포트 사용
        from src.api.client import MistralAPIClient
        self.client = MistralAPIClient()
        
        # 프롬프트 설정 로드
        self.prompts = self._load_yaml("configs/prompts.yaml")
        
        # 1. 전처리기 초기화
        data_cfg = config.get('data', {})
        self.preprocessor = AdaptiveFastPreprocessor(
            self.client, 
            self.prompts, 
            max_tokens=data_cfg.get('chunk_size', 3000),
            overlap=data_cfg.get('overlap', 300)
        )
        
        # 2. 로거 및 메타 정보 설정
        log_cfg = config.get('logging', {})
        self.logger = ExperimentLogger(base_path=log_cfg.get('base_path', 'results'))
        self.exp_name = config.get('experiment_name', 'v3_adaptive_packing')

        # 3. 검색 및 합성 전략 파라미터
        strat_cfg = config.get('strategy', {})
        self.landmark_top_n = strat_cfg.get('scan_top_n', 80)
        self.evidence_top_k = strat_cfg.get('final_top_k', 32)
        
        # [중요] 토큰 폭발 방지를 위한 가드레일: 원문(Raw Detail)은 상위 N개만 포함
        self.max_raw_details = 8 
        
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """YAML 설정 파일을 로드합니다."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_cache_path(self, file_name: str) -> str:
        """캐싱을 위한 안전한 파일 경로를 생성합니다."""
        safe_name = "".join([c if c.isalnum() else "_" for c in file_name])
        return os.path.join(self.cache_dir, f"{safe_name}_v3.pkl")

    def _get_embeddings(self, chunks: List[Dict[str, Any]], file_name: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """임베딩을 생성하거나 로컬 캐시에서 불러옵니다."""
        cache_path = self._get_cache_path(file_name)
        
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data["embeddings"], data["chunks"]

        print(f"[*] 임베딩 생성 중... ({len(chunks)} 청크)")
        texts = [c['content'] for c in chunks]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SentenceTransformer를 이용한 벡터화 (Batch 처리)
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
        """Planner를 통해 탐색 범위(TOP_N)와 전략적 계획을 수립합니다."""
        response = self.client.chat_completion([
            {"role": "system", "content": self.prompts['planner']['system']},
            {"role": "user", "content": self.prompts['planner']['user'].format(query=query)}
        ], temperature=0.0)
        
        top_n = self.landmark_top_n
        if "TOP_N:" in response:
            try:
                # 정규표현식을 사용하여 숫자 추출
                top_n = int(re.search(r'\d+', response.split("TOP_N:")[1]).group())
            except Exception:
                pass
        return top_n, response.strip()

    def run_pipeline(self, raw_text: str, query: str, file_name: str) -> str:
        """전체 RAG 파이프라인을 실행합니다."""
        start_time = time.time()
        
        try:
            # 1. 문서 스타일 분석 및 전처리
            # 캐시가 있으면 임베딩 함수에서 처리하므로 전처리는 스킵 가능
            cache_path = self._get_cache_path(file_name)
            if not os.path.exists(cache_path):
                chunks = self.preprocessor.preprocess(raw_text, {"file": file_name})
            else:
                chunks = [] # 캐시 로드 대기

            # 2. 임베딩 데이터 획득 및 플래닝
            embeddings, chunks = self._get_embeddings(chunks, file_name)
            dynamic_k, plan = self._create_dynamic_plan(query)
            print(f"[*] 동적 탐색 계획 수립: {dynamic_k}개 지점 정밀 분석 예정")

            # 3. 벡터 유사도 기반 후보군(Landmarks) 선정
            query_emb = self.client.st_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, torch.from_numpy(embeddings))[0].cpu().numpy()
            
            # 상위 K개 및 문서 시작/끝 지점 추가 (맥락 파악용)
            top_indices = np.argsort(-scores)[:dynamic_k]
            terminals = set(range(0, min(5, len(chunks)))) | set(range(max(0, len(chunks)-5), len(chunks)))
            scout_indices = sorted(list(set(top_indices) | terminals))

            # 4. 에이전트 스카우팅 (Navigator Phase)
            scout_reports = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_idx = {}
                for idx in scout_indices:
                    chunk = chunks[idx]
                    pos_info = f"{(idx/len(chunks))*100:.1f}% 지점"
                    
                    f = executor.submit(self.client.chat_completion, [
                        {"role": "system", "content": self.prompts['navigator']['system']},
                        {"role": "user", "content": self.prompts['navigator']['user'].format(
                            plan=plan, 
                            query=query, 
                            content=chunk['content'][:3500],
                            position_info=pos_info, 
                            chunk_index=idx, 
                            total_chunks=len(chunks)
                        )}
                    ], temperature=0.0)
                    future_to_idx[f] = idx
                
                for f in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(scout_indices), desc="Scouting"):
                    report_text = f.result()
                    idx = future_to_idx[f]
                    
                    # 보고서에서 VALUE_SCORE 추출
                    score = 5
                    if "VALUE_SCORE:" in report_text:
                        try:
                            score = int(re.search(r'\d+', report_text.split("VALUE_SCORE:")[1]).group())
                        except:
                            pass
                    
                    scout_reports.append({
                        "score": score,
                        "findings": report_text,
                        "raw_content": chunks[idx]['content'],
                        "pos": f"{(idx/len(chunks))*100:.1f}%"
                    })

            # 5. Adaptive Evidence Packing (Context Length 최적화 핵심)
            # 가치 점수가 높은 순으로 정렬하여 상위 TOP_K 선별
            sorted_reports = sorted(scout_reports, key=lambda x: x['score'], reverse=True)[:self.evidence_top_k]
            
            packed_evidence = ""
            for i, rep in enumerate(sorted_reports):
                # 모든 결과에 대해 Navigator의 분석 요약(FINDINGS)은 포함 (가볍고 핵심적)
                packed_evidence += f"\n[Evidence {i+1} at {rep['pos']} - Score: {rep['score']}]\n{rep['findings']}\n"
                
                # 상위 8개(max_raw_details)에 한해서만 모델이 직접 대조할 '상세 원문' 포함
                if i < self.max_raw_details:
                    packed_evidence += f"Detailed Excerpt: {rep['raw_content'][:1500]}\n"

            # 6. 최종 답변 합성 (Executor Phase)
            final_answer = self.client.chat_completion([
                {"role": "system", "content": self.prompts['synthesis']['system']},
                {"role": "user", "content": self.prompts['synthesis']['user'].format(
                    query=query, 
                    plan=plan, 
                    evidence=packed_evidence
                )}
            ])

            # 7. 결과 로깅 및 반환
            self.logger.log_result(self.exp_name, query, final_answer, start_time, [file_name])
            return final_answer

        except Exception as e:
            error_msg = f"Pipeline Error: {str(e)}"
            self.logger.log_result(self.exp_name, query, error_msg, start_time, [file_name])
            print(f"[!] 파이프라인 에러 발생: {e}")
            return error_msg