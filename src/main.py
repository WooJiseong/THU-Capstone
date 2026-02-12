import yaml
import os
from typing import Dict, Any, List
from src.core.orchestrator import WorkflowOrchestrator

def load_yaml(path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 안전하게 로드합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main() -> None:
    """
    실험 설정 및 시나리오를 로드하여 전체 파이프라인을 가동합니다.
    파일 내용을 직접 읽는 대신 경로 리스트를 오케스트레이터로 전달합니다.
    """
    print("=== Long Context LLM Experiment Runner (v4.5) ===")
    
    try:
        # 시스템 기본 설정 및 쿼리 시나리오 로드
        sys_config = load_yaml("configs/experiment_config.yaml")
        query_scenarios = load_yaml("configs/query_txt_config.yaml")
    except Exception as e:
        print(f"[!] 설정 파일 로드 실패: {e}")
        return

    # 실험 시나리오 반복 수행
    experiments: List[Dict[str, Any]] = query_scenarios.get('experiments', [])
    
    for exp in experiments:
        exp_name = exp.get('name', 'Unnamed Experiment')
        source_files: List[str] = exp.get('source_files', [])
        
        # 실제 존재하는 파일들만 필터링 (오류 방지)
        valid_paths = [p for p in source_files if os.path.exists(p)]
        if not valid_paths:
            print(f"\n[Skip] 유효한 파일이 없습니다: {exp_name}")
            continue

        print(f"\n[실험 시작] Experiment: {exp_name}")
        print(f"[*] 대상 파일: {', '.join([os.path.basename(p) for p in valid_paths])}")

        # 오케스트레이터 설정 및 초기화
        current_config = sys_config.copy()
        current_config['experiment_name'] = exp_name
        orchestrator = WorkflowOrchestrator(current_config)
        
        # 해당 실험의 쿼리들 순차 실행
        queries: List[str] = exp.get('queries', [])
        for query in queries:
            print(f"\n  [질문 처리 중] {query[:60]}...")
            try:
                # 오케스트레이터에 파일 경로 리스트와 쿼리 전달
                answer = orchestrator.run_pipeline(
                    file_paths=valid_paths,
                    query=query
                )
                print(f"  [답변 완료] Latency는 로그를 참조하십시오.")
            except Exception as e:
                print(f"  [에러 발생] 쿼리 처리 실패: {e}")

    print("\n=== 모든 실험이 완료되었습니다. 결과 폴더를 확인하세요. ===")

if __name__ == "__main__":
    main()