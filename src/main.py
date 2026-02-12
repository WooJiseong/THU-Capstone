import yaml
import os
from typing import Dict, Any, List
from src.core.orchestrator import WorkflowOrchestrator

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("=== Long Context LLM Experiment Runner v3 ===")
    try:
        sys_config = load_yaml("configs/experiment_config.yaml")
        query_scenarios = load_yaml("configs/query_txt_config.yaml")
    except Exception as e:
        print(f"[!] 설정 로드 실패: {e}")
        return

    for exp in query_scenarios.get('experiments', []):
        exp_name = exp.get('name', 'Unnamed')
        print(f"\n[Experiment] {exp_name}")
        combined_text = ""
        source_names = []
        for file_path in exp.get('source_files', []):
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n\n---\n\n"
                source_names.append(os.path.basename(file_path))
        
        if not combined_text: continue
        
        current_config = sys_config.copy()
        current_config['experiment_name'] = exp_name
        orchestrator = WorkflowOrchestrator(current_config)
        
        for query in exp.get('queries', []):
            print(f"  [Query] {query[:50]}...")
            orchestrator.run_pipeline(combined_text, query, ", ".join(source_names))

if __name__ == "__main__":
    main()