import time
import numpy as np
import concurrent.futures
from typing import List, Dict, Any
from sentence_transformers import util
from src.api.client import MistralAPIClient
from src.data_loader.preprocessor import GeneralMarkdownPreprocessor
from src.utils.logger import ExperimentLogger

class WorkflowOrchestrator:
    """
    범용적인 문서 구조 파악과 세부 데이터 추출을 결합한 
    '자가 구조화 하이브리드' 오케스트레이터입니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = MistralAPIClient()
        self.preprocessor = GeneralMarkdownPreprocessor(
            max_tokens=config['data'].get('chunk_size', 16000),
            overlap=config['data'].get('overlap', 1500)
        )
        self.logger = ExperimentLogger()
        self.exp_name = config.get('experiment_name', 'general_hybrid_exp')
        self.top_k = config.get('strategy', {}).get('top_k', 5)

    def _get_structural_landmark(self, chunk: Dict[str, Any]) -> str:
        """
        [Map] 문서의 도메인에 상관없이 구조적 이정표와 핵심 요소를 추출합니다.
        """
        # 특정 단어(Article 등)를 언급하지 않고 구조적 특징을 뽑아내도록 유도
        map_prompt = (
            "Analyze this document fragment and identify its structural position and key landmarks. "
            "1. List any sequential identifiers (numbers, dates, versions, or step labels).\n"
            "2. Identify the main topic or objective of this section.\n"
            "3. Note the very beginning and the very end markers of this fragment.\n\n"
            "Fragment:\n{content}"
        )
        return self.client.chat_completion([
            {"role": "user", "content": map_prompt.format(content=chunk['content'])}
        ], temperature=0.0, max_tokens=250)

    def _generate_landmarks_parallel(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """병렬 처리를 통해 문서 전체의 이정표 지도를 생성합니다."""
        print(f"[*] 분석 시작: {len(chunks)}개의 청크에서 구조적 이정표를 추출합니다...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            landmarks = list(executor.map(self._get_structural_landmark, chunks))
        return landmarks

    def _retrieve_relevant_content(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """질문과 의미론적으로 가장 유사한 원본 데이터를 추출합니다."""
        query_emb = self.client.st_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        chunk_contents = [c['content'] for c in chunks]
        chunk_embs = self.client.st_model.encode(chunk_contents, convert_to_tensor=True, show_progress_bar=False)
        
        cos_scores = util.cos_sim(query_emb, chunk_embs)[0]
        top_results = np.argsort(-cos_scores.cpu().numpy())[:self.top_k]
        
        return [chunks[int(i)] for i in top_results]

    def _synthesize_final_answer(self, query: str, landmarks: List[str], local_chunks: List[Dict[str, Any]]) -> str:
        """
        [Reduce] 전체 구조 지도(Landmarks)와 세부 증거(Local)를 결합하여 최종 답변을 생성합니다.
        """
        full_document_map = "\n".join([f"[Segment {i+1} Landmarks]: {m}" for i, m in enumerate(landmarks)])
        
        detailed_evidence = ""
        for i, lc in enumerate(local_chunks):
            detailed_evidence += f"---\n[Detailed Evidence {i+1}]\n{lc['content']}\n"

        # 모델에게 문서의 '전체성'을 고려하여 추론하도록 지시
        reduce_prompt = (
            "You are an expert analyst. You are provided with a 'Structural Map' of the entire document "
            "and 'Detailed Evidence' for specific parts.\n\n"
            "### [Structural Map of Entire Document]\n{document_map}\n\n"
            "### [Detailed Evidence from Specific Parts]\n{evidence}\n\n"
            "### [User Query]\n{query}\n\n"
            "Instructions:\n"
            "1. If the query asks for a count, list, or summary of the entire document, use the Structural Map "
            "to trace the sequence of identifiers across all segments.\n"
            "2. Use the Detailed Evidence to verify specific facts and provide precise quotes or details.\n"
            "3. Synthesize both sources to provide a logical, comprehensive answer. If a specific count "
            "is needed, ensure you account for the flow from the first to the last segment."
        )
        
        return self.client.chat_completion([
            {"role": "user", "content": reduce_prompt.format(
                query=query, 
                document_map=full_document_map, 
                evidence=detailed_evidence
            )}
        ])

    def run_pipeline(self, raw_text: str, query: str, file_name: str) -> str:
        start_time = time.time()
        chunks = self.preprocessor.preprocess(raw_text, {"file_name": file_name})
        
        if not chunks:
            return "Error: 전처리 결과가 비어있습니다."

        # Step 1: 자가 구조화 맵 생성 (Global)
        landmarks = self._generate_landmarks_parallel(chunks)
        
        # Step 2: 유사도 기반 검색 (Local)
        local_evidence = self._retrieve_relevant_content(query, chunks)
        
        # Step 3: 종합 추론 (Reduce)
        final_answer = self._synthesize_final_answer(query, landmarks, local_evidence)

        self.logger.log_result(self.exp_name, query, final_answer, start_time, [file_name])
        return final_answer