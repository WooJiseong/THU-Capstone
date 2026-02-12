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
    [Enterprise Grade] RAG 파이프라인의 에이전트 워크플로우를 조율합니다.
    - Adaptive Strategy: 쿼리 성격에 따라 탐색(N) 및 채택(K) 범위를 유동적으로 조절
    - Scalability: 최대 4,000페이지 분량의 PDF 처리를 고려한 병렬 스카우팅 및 캐싱
    - User Experience: 직관적인 진행 상황 모니터링 및 단계별 로깅 강화
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API 클라이언트 지연 임포트 (순환 참조 방지 및 초기화 속도 개선)
        from src.api.client import MistralAPIClient
        self.client = MistralAPIClient()
        
        # 프롬프트 설정 로드
        self.prompts = self._load_yaml("configs/prompts.yaml")
        
        # 전처리기 초기화
        data_cfg = config.get('data', {})
        self.preprocessor = AdaptiveFastPreprocessor(
            self.client, 
            self.prompts, 
            max_tokens=data_cfg.get('chunk_size', 3000),
            overlap=data_cfg.get('overlap', 300)
        )
        
        # 로깅 및 실험 관리
        log_cfg = config.get('logging', {})
        self.logger = ExperimentLogger(base_path=log_cfg.get('base_path', 'results'))
        self.exp_name = config.get('experiment_name', 'v4_adaptive_topk_pipeline')

        # 기본 전략 파라미터
        strat_cfg = config.get('strategy', {})
        self.default_top_n = strat_cfg.get('scan_top_n', 100)
        self.default_top_k = strat_cfg.get('final_top_k', 24)
        
        self.max_raw_details = 6 
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """YAML 설정 파일을 안전하게 로드합니다."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _generate_cache_key(self, file_paths: List[str]) -> str:
        """파일 경로 기반 유니크 캐시 키 생성"""
        combined_string = "|".join(sorted(file_paths))
        hash_val = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
        base_name = os.path.basename(file_paths[0])[:15]
        safe_prefix = "".join([c if c.isalnum() else "_" for c in base_name])
        return f"{safe_prefix}_{hash_val}_v4_adaptive"

    def _get_embeddings(self, chunks: List[Dict[str, Any]], cache_key: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """임베딩 벡터 획득 (캐시 우선순위)"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            print(f"    >>> [캐시 발견] 이전 임베딩 데이터를 로드합니다.")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data["embeddings"], data["chunks"]

        if not chunks:
            raise ValueError(f"데이터가 없으며 캐시도 존재하지 않습니다: {cache_key}")

        print(f"    >>> [임베딩 생성] {len(chunks)}개 청크에 대한 벡터화 진행 중...")
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

    def _create_adaptive_strategy(self, query: str) -> Tuple[int, int, str]:
        """Mistral을 이용한 동적 탐색 계획 수립"""
        response = self.client.chat_completion([
            {"role": "system", "content": self.prompts['planner']['system']},
            {"role": "user", "content": self.prompts['planner']['user'].format(query=query)}
        ], temperature=0.0)
        
        top_n = self.default_top_n
        top_k = self.default_top_k
        
        try:
            n_match = re.search(r'TOP_N:\s*(\d+)', response)
            k_match = re.search(r'TOP_K:\s*(\d+)', response)
            if n_match: top_n = int(n_match.group(1))
            if k_match: top_k = int(k_match.group(1))
            top_k = min(top_k, top_n)
        except:
            pass

        return top_n, top_k, response.strip()

    def run_pipeline(self, file_paths: List[str], query: str) -> str:
        """전체 RAG 워크플로우 실행 및 시각적 진행 상황 출력"""
        print(f"\n{'='*60}\n[RUN] 파이프라인 실행 시작\n쿼리: {query}\n{'='*60}")
        pipeline_start = time.time()
        cache_key = self._generate_cache_key(file_paths)
        
        try:
            # 1. 데이터 전처리
            step_start = time.time()
            print(f"\n[STEP 1/8] 데이터 전처리 및 캐시 확인 중...")
            all_chunks = []
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if not os.path.exists(cache_path):
                for path in file_paths:
                    print(f"    >>> 파일 분석 중: {os.path.basename(path)}")
                    file_chunks = self.preprocessor.preprocess(path, {"source": os.path.basename(path)})
                    all_chunks.extend(file_chunks)
                chunks = all_chunks
            else:
                chunks = []
            print(f"[DONE] 전처리 완료 ({time.time()-step_start:.2f}s)")

            # 2. 임베딩 및 전략 수립
            step_start = time.time()
            print(f"\n[STEP 2/8] 벡터 임베딩 생성 및 동적 전략 수립 중...")
            embeddings, chunks = self._get_embeddings(chunks, cache_key)
            dynamic_n, dynamic_k, plan = self._create_adaptive_strategy(query)
            print(f"    >>> 동적 전략 결과: 탐색 범위(N)={dynamic_n}, 추출 범위(K)={dynamic_k}")
            print(f"[DONE] 전략 수립 완료 ({time.time()-step_start:.2f}s)")

            # 3. 벡터 유사도 검색
            step_start = time.time()
            print(f"\n[STEP 3/8] 벡터 유사도 기반 후보 청크 탐색 중...")
            query_emb = self.client.st_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_emb, torch.from_numpy(embeddings))[0].cpu().numpy()
            top_indices = np.argsort(-scores)[:dynamic_n]
            
            terminals = set(range(0, min(5, len(chunks)))) | set(range(max(0, len(chunks)-5), len(chunks)))
            scout_indices = sorted(list(set(top_indices) | terminals))
            print(f"    >>> 전체 {len(chunks)}개 중 {len(scout_indices)}개의 후보군 선별 완료")
            print(f"[DONE] 후보 탐색 완료 ({time.time()-step_start:.2f}s)")

            # 4. Navigator 스카우팅
            step_start = time.time()
            print(f"\n[STEP 4/8] 개별 청크 정밀 분석(Scouting) 진행 중 (병렬 처리)...")
            scout_reports = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_idx = {}
                for idx in scout_indices:
                    chunk = chunks[idx]
                    pos_info = f"{(idx/len(chunks))*100:.1f}% 지점"
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
            print(f"[DONE] 정밀 분석 완료 ({time.time()-step_start:.2f}s)")

            # 5. Evidence Packing
            step_start = time.time()
            print(f"\n[STEP 5/8] 고가치 정보 선별 및 증거 패키징 중...")
            sorted_reports = sorted(scout_reports, key=lambda x: x['score'], reverse=True)[:dynamic_k]
            packed_evidence = ""
            for i, rep in enumerate(sorted_reports):
                packed_evidence += f"\n[Evidence {i+1} from {rep['source']}]\n{rep['findings']}\n"
                if i < self.max_raw_details:
                    packed_evidence += f"Detailed Excerpt: {rep['raw_content'][:2000]}\n"
            print(f"    >>> 상위 {len(sorted_reports)}개의 증거가 최종 합성에 포함됩니다.")
            print(f"[DONE] 패키징 완료 ({time.time()-step_start:.2f}s)")

            # 6. Technical Synthesis
            step_start = time.time()
            print(f"\n[STEP 6/8] 기술 보고서 합성 중 (Mistral-Small)...")
            technical_report = self.client.chat_completion([
                {"role": "system", "content": self.prompts['synthesis']['system']},
                {"role": "user", "content": self.prompts['synthesis']['user'].format(
                    query=query, evidence=packed_evidence
                )}
            ])
            print(f"[DONE] 합성 완료 ({time.time()-step_start:.2f}s)")

            # 7. Response Refinement
            step_start = time.time()
            print(f"\n[STEP 7/8] 사용자 친화적 응답 정제 중...")
            final_answer = self.client.chat_completion([
                {"role": "system", "content": self.prompts['refiner']['system']},
                {"role": "user", "content": self.prompts['refiner']['user'].format(
                    query=query, report=technical_report
                )}
            ])
            print(f"[DONE] 정제 완료 ({time.time()-step_start:.2f}s)")

            # 8. 최종 결과 반환
            print(f"\n[STEP 8/8] 처리 완료 및 결과 로깅 중...")
            self.logger.log_result(self.exp_name, query, final_answer, pipeline_start, file_paths)
            
            total_time = time.time() - pipeline_start
            print(f"\n{'='*60}\n[SUCCESS] 전체 처리 시간: {total_time:.2f}s\n{'='*60}\n")
            return final_answer

        except Exception as e:
            error_msg = f"Critical Pipeline Error: {str(e)}"
            self.logger.log_result(self.exp_name, query, error_msg, pipeline_start, file_paths)
            print(f"\n[!] 에러 발생: {e}")
            return error_msg