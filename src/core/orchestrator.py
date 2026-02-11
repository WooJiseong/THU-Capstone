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
    Enhanced RAPTOR-inspired Two-Way Orchestrator.
    Fuses structured global context with high-fidelity raw tokens using 
    explicit delimiters and multi-step reasoning prompts.
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
        self.exp_name = config.get('experiment_name', 'structured_hybrid_v2')
        
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

        print(f"[*] Vectorizing {len(chunks)} chunks for semantic retrieval...")
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

    def _create_scout_landmark(self, chunk: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        [Step 1: Global Scouting] 
        Produces a structured summary of the segment's role and boundaries.
        """
        landmark_prompt = (
            "### TASK: STRUCTURAL LANDMARK EXTRACTION\n"
            "Analyze the following segment to provide context for the query: '{query}'\n\n"
            "--- SEGMENT START ---\n{content}\n--- SEGMENT END ---\n\n"
            "Respond in the following format:\n"
            "RELEVANCE_SCORE: [0-10]\n"
            "SEQ_MARKERS: [List all sequential IDs like Article numbers, Version tags, or Dates]\n"
            "TERMINAL_MARKER: [The very last ID or number in this segment]\n"
            "CONTENT_BRIEF: [1-sentence technical summary]"
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
            "metadata": chunk['metadata']
        }

    def _synthesize_hybrid_results(self, query: str, scout_results: List[Dict[str, Any]], total_chunks: int) -> str:
        """
        [Step 3: Purpose-Driven Synthesis] 
        Fuses Global Map and Raw Evidence using distinct delimiters.
        """
        # Select best raw evidence
        sorted_results = sorted(scout_results, key=lambda x: x['score'], reverse=True)
        evidence_chunks = [r for r in sorted_results[:self.evidence_top_k] if r['score'] >= 4]
        
        # Build generalized structural map
        map_indices = set()
        map_indices.add(0) # Document Head
        map_indices.add(total_chunks - 1) # Document Tail
        for i, res in enumerate(scout_results):
            if res['score'] >= 7: map_indices.add(i)
        
        sample_rate = max(1, total_chunks // 10)
        for i in range(0, total_chunks, sample_rate): map_indices.add(i)

        sorted_map_indices = sorted(list(map_indices))
        map_entries = []
        for idx in sorted_map_indices:
            res = scout_results[idx]
            map_entries.append(f"<CHUNK_POSITION index='{idx+1}/{total_chunks}'>\n{res['landmark']}\n</CHUNK_POSITION>")
        
        global_map_str = "\n".join(map_entries)
        
        evidence_blocks = []
        for i, ev in enumerate(evidence_chunks):
            loc = " > ".join(ev['metadata'].get('breadcrumbs', ['Root']))
            evidence_blocks.append(
                f"<RAW_EVIDENCE_BLOCK source='{loc}' rank='{i+1}'>\n{ev['content']}\n</RAW_EVIDENCE_BLOCK>"
            )
        detailed_evidence_str = "\n".join(evidence_blocks)

        final_prompt = (
            "### ROLE: SENIOR TECHNICAL AUDITOR\n"
            "You are provided with two distinct layers of information to answer the query: '{query}'\n\n"
            
            "--- LAYER 1: GLOBAL_STRUCTURE_MAP ---\n"
            "Use this to understand the sequence, flow, and boundaries of the entire document.\n"
            "<GLOBAL_STRUCTURE_MAP>\n{global_map}\n</GLOBAL_STRUCTURE_MAP>\n\n"
            
            "--- LAYER 2: TARGETED_RAW_EVIDENCE ---\n"
            "Use this for high-precision extraction of facts, technical specs, and literal quotes.\n"
            "<TARGETED_RAW_EVIDENCE>\n{evidence}\n</TARGETED_RAW_EVIDENCE>\n\n"
            
            "### REASONING PROTOCOL:\n"
            "1. **Analyze Sequence**: Trace the progression of identifiers (Articles, Versions, IDs) in the GLOBAL_STRUCTURE_MAP.\n"
            "2. **Find Terminal Points**: To identify the total count or the final version, look at the very last CHUNK_POSITION entry.\n"
            "3. **Extract Facts**: Cross-reference the identifiers with the TARGETED_RAW_EVIDENCE to find the specific values required.\n"
            "4. **Synthesize**: Merge the structural context with the raw data. If a table or list is requested, format it clearly.\n"
            "5. **Verify**: Ensure the answer is strictly derived from the provided tags. Cite chunk positions for traceability.\n\n"
            "FINAL RESPONSE:"
        )
        
        return self.client.chat_completion([
            {"role": "user", "content": final_prompt.format(
                query=query, global_map=global_map_str, evidence=detailed_evidence_str
            )}
        ], max_tokens=2500)

    def run_pipeline(self, raw_text: str, query: str, file_name: str) -> str:
        start_time = time.time()
        
        # Cache-aware Preprocessing
        text_hash = hash(raw_text)
        if self._doc_cache["text_hash"] == text_hash and self._doc_cache["chunks"]:
            chunks = self._doc_cache["chunks"]
        else:
            chunks = self.preprocessor.preprocess(raw_text, {"file_name": file_name})
        
        print(f"[*] Initiating structured hybrid analysis: {file_name} ({len(chunks)} chunks)")

        # 1. Similarity-based Pruning
        embeddings = self._get_embeddings(chunks, raw_text)
        query_emb = self.client.st_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
        
        top_indices = np.argsort(-scores)[:self.landmark_top_n]
        all_scout_results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
        scout_indices = set(top_indices) | {0, len(chunks) - 1}
        
        # 2. Parallel Scouting (The "Forest" Layer)
        print(f"[*] Scouting {len(scout_indices)} key structural landmarks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {executor.submit(self._create_scout_landmark, chunks[idx], query): idx for idx in scout_indices}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                all_scout_results[idx] = future.result()

        # Fill metadata for unscouted chunks to preserve document map integrity
        final_scout_results: List[Dict[str, Any]] = []
        for i in range(len(chunks)):
            res = all_scout_results[i]
            if res is None:
                loc = " > ".join(chunks[i]['metadata'].get('breadcrumbs', ['Unknown Section']))
                final_scout_results.append({
                    "landmark": f"RELEVANCE_SCORE: {round(float(scores[i])*10, 1)}\nSEQ_MARKERS: N/A\nCONTENT_BRIEF: Static marker for section {loc}",
                    "score": float(scores[i]) * 4,
                    "content": chunks[i]['content'],
                    "metadata": chunks[i]['metadata']
                })
            else:
                final_scout_results.append(res)

        # 3. Final Synthesis (The Fusion Layer)
        final_answer = self._synthesize_hybrid_results(query, final_scout_results, len(chunks))
        self.logger.log_result(self.exp_name, query, final_answer, start_time, [file_name])
        return final_answer