import re
import multiprocessing
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from tqdm import tqdm

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass

class FastLongContextPreprocessor(BasePreprocessor):
    """
    [진행도 표시 및 강제 병렬화 기능이 통합된 고성능 전처리기]
    """
    def __init__(self, max_tokens: int = 5000, overlap: int = 1000):
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap * 4
        # 마크다운 헤더 패턴 (헤더 뒤에 공백이 있는 표준 규격 기준)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.ref_pattern = re.compile(
            r'(\(?(?:Table|Figure|p\.)\s?\d+(?:-\d+)?\)?|see|reference)', 
            re.IGNORECASE
        )
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _split_by_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        matches = list(self.header_pattern.finditer(text))
        if not matches:
            # 헤더가 없을 경우 루트 섹션 하나로 반환
            return [{"level": 1, "title": "Root", "content": text}]
            
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            sections.append({
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "content": text[start:end].strip()
            })
        return sections

    def _force_split_text(self, text: str) -> List[Dict[str, Any]]:
        total_len = len(text)
        # 최소한 max_chars 보다는 크게 쪼개지도록 설정
        split_size = max(self.max_chars * 2, total_len // self.num_workers)
        
        forced_sections = []
        start = 0
        idx = 1
        
        while start < total_len:
            end = start + split_size
            # 마지막 조각 처리 및 문장 끊김 방지를 위한 여백 허용
            if end > total_len:
                end = total_len
            
            forced_sections.append({
                "level": 1,
                "title": f"Auto-Segment {idx}",
                "content": text[start:end]
            })
            start = end
            idx += 1
            
        return forced_sections

    def _recursive_split(self, text: str) -> List[str]:
        """문장 경계를 보존하며 재귀적으로 텍스트 분할"""
        chunks = []
        start = 0
        limit = self.max_chars
        overlap = self.overlap_chars
        
        while start < len(text):
            end = start + limit
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            chunk_candidate = text[start:end]
            split_idx = -1
            # 구조적 분할점 탐색
            for separator in ['\n\n', '\n', '. ', ' ']:
                last_idx = chunk_candidate.rfind(separator)
                if last_idx != -1:
                    split_idx = last_idx + len(separator)
                    break
            
            actual_end = start + split_idx if split_idx != -1 else end
            chunks.append(text[start:actual_end])
            # Overlap 적용하여 다음 시작점 설정
            start = max(start + 1, actual_end - overlap)
            
        return chunks

    def _extract_references(self, text: str) -> List[str]:
        """텍스트 내 상호 참조 정보(표, 그림 등) 추출"""
        return list(set(self.ref_pattern.findall(text)))

    def _process_section_unit(self, section: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Worker 프로세스에서 실행될 독립 작업 단위"""
        content = section['content']
        title = section['title']
        
        # 실제 청킹 로직 수행
        sub_contents = self._recursive_split(content) if len(content) > self.max_chars else [content]
        
        results = []
        for sc in sub_contents:
            refs = self._extract_references(sc)
            results.append({
                "content": f"[Location: {title}]\n{sc}",
                "metadata": {
                    **metadata,
                    "section_title": title,
                    "references": refs,
                    "char_count": len(sc)
                }
            })
        return results

    def preprocess(self, raw_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        4,000페이지 문서를 병렬로 빠르게 처리하여 청크 리스트를 생성합니다.
        """
        print("[*] Step 1: 텍스트 정제 및 구조 분석 중...")
        cleaned_text = self._clean_text(raw_text)
        sections = self._split_by_hierarchy(cleaned_text)
        
        # [핵심] 섹션이 너무 적으면 강제로 분할하여 병렬 효율 극대화
        if len(sections) == 1 and len(cleaned_text) > self.max_chars:
            print(f"[*] 알림: 단일 섹션 감지. 성능 향상을 위해 {self.num_workers}개 단위로 강제 분할합니다.")
            sections = self._force_split_text(cleaned_text)
        
        total_sections = len(sections)
        all_chunks = []
        
        print(f"[*] Step 2: 병렬 전처리 가동 (단위: {total_sections} 섹션 / Workers: {self.num_workers})")

        # 
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 섹션 단위로 병렬 작업 제출
            futures = [executor.submit(self._process_section_unit, s, metadata) for s in sections]
            
            # tqdm 진행 바 표시
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_sections, desc="Processing"):
                try:
                    all_chunks.extend(future.result())
                except Exception as e:
                    print(f"\n[Error] 세그먼트 처리 중 오류 발생: {e}")

        print(f"[*] Step 3: 완료 (생성된 청크 수: {len(all_chunks)})")
        return all_chunks