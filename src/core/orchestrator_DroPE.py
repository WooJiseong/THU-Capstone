#DroPE
import time
import numpy as np
import concurrent.futures
from typing import List, Dict, Any, Optional
from sentence_transformers import util
from src.api.client import MistralAPIClient
from src.data_loader.preprocessor import GeneralMarkdownPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    Location-Aware RAPTOR Orchestrator (Inspired by DroPE concepts).
    Implements discrete position marking to enhance robustness against 'Lost in the Middle'
    and ensures accurate terminal identifier detection for aggregation queries.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = MistralAPIClient()
        
        data_cfg = config.get('data', {})
        self.preprocessor = GeneralMarkdownPreprocessor(
            max_tokens=data_cfg.get('chunk_size', 5000),
            overlap=data_cfg.get('overlap', 1000)
        )
        
        self.logger = ExperimentLogger()
        self.exp_name = config.get('experiment_name', 'drope_inspired_v1')
        
        strat_cfg = config.get('strategy', {})
        self.landmark_top_n = strat_cfg.get('scan_top_n', 50)
        self.evidence_top_k = strat_cfg.get('final_top_k', 12)
        
        self._doc_cache: Dict[str, Any] = {
            "text_hash": None,
            "embeddings": None,
            "chunks": None
        }

    def _get_embeddings(self, chunks: List[Dict[str, Any]], raw_text: str) -> np.ndarray:
        text_hash = hash(raw_text)
        if self._doc_cache["text_hash"] == text_hash and self._doc_cache["embeddings"] is not None:
            return self._doc_cache["embeddings"]

        print(f"[*] 벡터화 수행 중: {len(chunks)} 청크 분석...")
        chunk_contents = [c['content'] for c in chunks]
        embeddings = self.client.st_model.encode(
            chunk_contents, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            batch_size=64
        )
        
        emb_np = embeddings.cpu().numpy()
        self._doc_cache.update({"text_hash": text_hash, "embeddings": emb_np, "chunks": chunks})
        return emb_np

    def _create_scout_landmark(self, chunk: Dict[str, Any], query: str, pos_idx: int, total: int) -> Dict[str, Any]:
        """
        [Step 1: Global Scouting] 
        각 섹션의 위치(Discrete Position)와 구조적 이정표를 추출합니다.
        """
        landmark_prompt = (
            f"### POSITION_CONTEXT: Segment {pos_idx}/{total}\n"
            "### TASK: STRUCTURAL SCOUTING\n"
            "Analyze the following segment to find markers for: '{query}'\n\n"
            "--- CONTENT START ---\n{content}\n--- CONTENT END ---\n\n"
            "Respond in the following structured format:\n"
            "RELEVANCE_SCORE: [0-10]\n"
            "SEQ_MARKERS: [List all Articles/Sections/Versions found]\n"
            "TERMINAL_NUMBER: [The highest or final sequential number here]\n"
            "POSITIONAL_ROLE: [e.g., Introduction, Core Regulation, Final Provisions]"
        )
        
        response = self.client.chat_completion([
            {"role": "user", "content": landmark_prompt.format(query=query, content=chunk['content'][:3500])}
        ], temperature=0.0, max_tokens=250)
        
        score = 0
        if "RELEVANCE_SCORE:" in response:
            try:
                score_part = response.split("RELEVANCE_SCORE:")[1].split("\n")[0].strip()
                score = int(''.join(filter(str.isdigit, score_part)))
            except: score = 5
            
        return {
            "landmark": response,
            "score": score,
            "content": chunk['content'],
            "metadata": chunk['metadata'],
            "pos_idx": pos_idx
        }

    def _synthesize_hybrid_results(self, query: str, scout_results: List[Dict[str, Any]], total_chunks: int) -> str:
        """
        [Step 3: Location-Aware Synthesis] 
        이산적 위치 인덱스(Discrete Position)를 사용하여 숲과 나무를 결합합니다.
        """
        sorted_results = sorted(scout_results, key=lambda x: x['score'], reverse=True)
        evidence_chunks = [r for r in sorted_results[:self.evidence_top_k] if r['score'] >= 4]
        
        # 맵 구성 (DroPE의 철학: 시작, 끝, 중간의 이산적 샘플링 보장)
        map_indices = set()
        map_indices.add(0) # First (Start)
        map_indices.add(total_chunks - 1) # Last (Terminal)
        
        for i, res in enumerate(scout_results):
            if res['score'] >= 7: map_indices.add(i)
        
        sample_rate = max(1, total_chunks // 10)
        for i in range(0, total_chunks, sample_rate): map_indices.add(i)

        sorted_map_indices = sorted(list(map_indices))
        map_entries = []
        for idx in sorted_map_indices:
            res = scout_results[idx]
            # XML 태그 내에 이산적 위치 정보를 명시적으로 포함
            map_entries.append(
                f"<LANDMARK_MAP_ENTRY position='{idx+1}' total='{total_chunks}' status='{'SAMPLED' if res.get('landmark').startswith('RELEVANCE') else 'DYNAMIC'}'>\n"
                f"{res['landmark']}\n"
                f"</LANDMARK_MAP_ENTRY>"
            )
        
        global_map_str = "\n".join(map_entries)
        
        evidence_blocks = []
        for i, ev in enumerate(evidence_chunks):
            loc = " > ".join(ev['metadata'].get('breadcrumbs', ['Root']))
            # 원문 데이터에도 위치 태그 부여
            evidence_blocks.append(
                f"<DETAILED_EVIDENCE_SOURCE rank='{i+1}' absolute_pos='{ev['pos_idx']+1}'>\n"
                f"SOURCE_PATH: {loc}\n"
                f"CONTENT: {ev['content']}\n"
                f"</DETAILED_EVIDENCE_SOURCE>"
            )
        detailed_evidence_str = "\n".join(evidence_blocks)

        final_prompt = (
            "### SYSTEM ROLE: DISCRETE POSITION-AWARE AUDITOR\n"
            "Your objective is to answer: '{query}' using hierarchical layers of document data.\n\n"
            
            "--- LAYER 1: GLOBAL_LANDMARK_MAP (The Context Forest) ---\n"
            "Provides discrete anchor points across the entire document timeline.\n"
            "<GLOBAL_LANDMARK_MAP>\n{global_map}\n</GLOBAL_LANDMARK_MAP>\n\n"
            
            "--- LAYER 2: RAW_EVIDENCE_DATA (The Specific Trees) ---\n"
            "Provides high-resolution raw tokens from prioritized locations.\n"
            "<RAW_EVIDENCE_DATA>\n{evidence}\n</RAW_EVIDENCE_DATA>\n\n"
            
            "### AUDIT PROTOCOL:\n"
            "1. **Analyze Sequential Flow**: Observe how identifiers (Article/Version) progress in the GLOBAL_LANDMARK_MAP.\n"
            "2. **Determine Termination**: To answer 'How many' or 'Final extent', you MUST examine the LANDMARK_MAP_ENTRY with position='{total_chunks}'.\n"
            "3. **Cross-Reference**: Match the terminal markers in the Map with the actual text in RAW_EVIDENCE_DATA.\n"
            "4. **Synthesize with Position Awareness**: Synthesize the final answer, ensuring that information from later positions is given priority for 'total' counts.\n"
            "5. **Validation**: Cite specific position indices (e.g., 'As seen in position X...') for every claim.\n\n"
            "FINAL AUDIT RESPONSE:"
        )
        
        return self.client.chat_completion([
            {"role": "user", "content": final_prompt.format(
                query=query, global_map=global_map_str, evidence=detailed_evidence_str, total_chunks=total_chunks
            )}
        ], max_tokens=2500)

    def run_pipeline(self, raw_text: str, query: str, file_name: str) -> str:
        start_time = time.time()
        
        text_hash = hash(raw_text)
        if self._doc_cache["text_hash"] == text_hash and self._doc_cache["chunks"]:
            chunks = self._doc_cache["chunks"]
        else:
            chunks = self.preprocessor.preprocess(raw_text, {"file_name": file_name})
        
        print(f"[*] 하이브리드 위치 인지 분석 시작: {file_name} ({len(chunks)} chunks)")

        embeddings = self._get_embeddings(chunks, raw_text)
        query_emb = self.client.st_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
        
        top_indices = np.argsort(-scores)[:self.landmark_top_n]
        all_scout_results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        scout_indices = set(top_indices) | {0, len(chunks) - 1}
        
        print(f"[*] Scouting {len(scout_indices)} key positions (including terminal boundaries)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # position index를 명시적으로 전달하여 DroPE 스타일의 위치 정보 부여
            future_to_idx = {executor.submit(self._create_scout_landmark, chunks[idx], query, idx, len(chunks)): idx for idx in scout_indices}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                all_scout_results[idx] = future.result()

        final_scout_results: List[Dict[str, Any]] = []
        for i in range(len(chunks)):
            res = all_scout_results[i]
            if res is None:
                loc = " > ".join(chunks[i]['metadata'].get('breadcrumbs', ['Section']))
                final_scout_results.append({
                    "landmark": f"RELEVANCE_SCORE: {round(float(scores[i])*10, 1)}\nSEQ_MARKERS: N/A\nPOSITIONAL_ROLE: Static Navigation Point ({loc})",
                    "score": float(scores[i]) * 4,
                    "content": chunks[i]['content'],
                    "metadata": chunks[i]['metadata'],
                    "pos_idx": i
                })
            else:
                final_scout_results.append(res)

        final_answer = self._synthesize_hybrid_results(query, final_scout_results, len(chunks))
        self.logger.log_result(self.exp_name, query, final_answer, start_time, [file_name])
        return final_answer