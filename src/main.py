import yaml
import os
from typing import Dict, Any
from src.core.orchestrator import WorkflowOrchestrator

def load_yaml(path: str) -> Dict[str, Any]:
    """YAML 파일을 안전하게 로드합니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("=== Long Context LLM Experiment Runner ===")
    
    # 1. 설정 로드
    try:
        sys_config = load_yaml("configs/experiment_config.yaml")
        query_scenarios = load_yaml("configs/query_txt_config.yaml")
    except Exception as e:
        print(f"Error loading configs: {e}")
        return

    # 2. 실험 시나리오 반복 수행
    for exp in query_scenarios.get('experiments', []):
        exp_name = exp.get('name', 'Unnamed Experiment')
        print(f"\n[Experiment] Starting: {exp_name}")

        # 소스 파일들의 텍스트 통합
        combined_text = ""
        source_names = []
        for file_path in exp.get('source_files', []):
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n\n---\n\n"
                source_names.append(os.path.basename(file_path))
            else:
                print(f"  [Warning] File not found: {file_path}")

        if not combined_text:
            print(f"  [Skip] No valid source content found for {exp_name}")
            continue

        # 오케스트레이터 설정 (시스템 설정 + 실험 이름 주입)
        current_config = sys_config.copy()
        current_config['experiment_name'] = exp_name
        
        orchestrator = WorkflowOrchestrator(current_config)
        
        # 3. 쿼리 실행
        file_label = ", ".join(source_names)
        for query in exp.get('queries', []):
            print(f"  [Query] Processing: {query[:50]}...")
            try:
                # 하이브리드 파이프라인 실행
                answer = orchestrator.run_pipeline(
                    raw_text=combined_text,
                    query=query,
                    file_name=file_label
                )
                # 결과는 Orchestrator 내부에서 Logger를 통해 CSV에 기록됨
            except Exception as e:
                print(f"  [Error] Failed to process query: {e}")

    print("\n=== All experiments completed. Check the results folder. ===")

if __name__ == "__main__":
    main()