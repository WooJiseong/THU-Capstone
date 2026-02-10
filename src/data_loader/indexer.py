import re
from typing import List, Dict, Any

class GlobalIndexer:
    """
    문서 내 날짜 / 버전 정보를 탐지하여 핵심 정보 구역을 인덱싱합니다.
    """
    def __init__(self):
        pass

    def build_structural_map(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        청크 리스트를 분석하여 타임라인 이벤트와 중요 정보 ID를 추출합니다.
        """
        timeline_events = []
        important_ids = []
        
        date_pattern = re.compile(r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b|\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b)", re.IGNORECASE)

        for chunk in chunks:
            content = chunk.get('content', '')
            cid = chunk.get('chunk_id', 0)
            
            dates = date_pattern.findall(content)
            if dates:
                important_ids.append(cid)
                for d in dates:
                    timeline_events.append({"date": d, "chunk_id": cid})

        print(f"[Indexer] Indexed {len(important_ids)} chunks containing factual dates and versions.")
        
        return {
            "important_ids": important_ids[:60], # 64k 윈도우 최적화를 위해 상위 60개 유지
            "timeline_events": timeline_events
        }