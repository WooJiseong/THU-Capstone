from typing import List, Dict, Any

class LongContextOrchestrator:
    """
    구조적 인덱스와 동적 키워드 매칭을 결합하여 최적의 64k 컨텍스트를 조립합니다.
    """
    def __init__(self, model_mgr):
        self.model_mgr = model_mgr

    def process_query(self, query: str, chunks: List[Dict[str, Any]], structural_map: Dict[str, Any]) -> str:
        query_lower = query.lower()
        
        # 1. Selection: 인덱싱된 중요 청크 + 쿼리 키워드 매칭
        selected_ids = set(structural_map.get('important_ids', []))
        
        # 질문 내 핵심 단어가 포함된 청크 동적 수집
        keywords = [w.strip() for w in query_lower.split() if len(w) > 3]
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            if any(kw in content_lower for kw in keywords):
                selected_ids.add(chunk['chunk_id'])
                # 컨텍스트 포화 방지를 위해 60개 내외로 제한
                if len(selected_ids) > 60: break

        # 2. Assembly: ID 순 정렬로 문서의 논리적 흐름 보존
        final_ids = sorted(list(selected_ids))
        context_parts = []
        current_chars = 0
        char_limit = 230000 # 64k 토큰을 고려한 안전한 문자수 제한

        for cid in final_ids:
            if cid >= len(chunks): continue
            chunk = chunks[cid]
            # 모델이 각 조각을 개별 데이터로 인식하게 태깅
            chunk_text = f"\n[FRAGMENT #{cid}]\n{chunk['content']}\n"
            
            if current_chars + len(chunk_text) > char_limit:
                break
            context_parts.append(chunk_text)
            current_chars += len(chunk_text)

        print(f"[Orchestrator] Context Assembled: {len(context_parts)} fragments ({current_chars} chars).")

        # 3. Prompting: LLM 분석에 특화된 지시문 구성
        prompt = f"""
        [INST]
### ROLE
You are a professional Technical Documentation Analyst. Your goal is to provide accurate, evidence-based answers based strictly on the provided context.

### CONTEXT
The following is a collection of technical documentation segments:
<context>
{"".join(context_parts)}
</context>

### INSTRUCTIONS
Based on the context provided above, answer the following question:
Question: {query}

### CONSTRAINTS
1. **Strict Fidelity:** Use only the information provided in the <context>. Do not use outside knowledge.
2. **Precision:** If the document mentions specific dates, versions, build numbers, or technical specifications, use them exactly as written.
3. **Honesty:** If the answer is not contained within the context or if the context is insufficient to provide a complete answer, state clearly that you do not know. Do not hallucinate or speculate.
4. **Language:** Provide the final answer in English (unless otherwise specified).
[/INST]
"""

        return self.model_mgr.generate(prompt)