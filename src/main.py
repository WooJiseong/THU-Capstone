import os
import sys
import time
import yaml  # pip install pyyaml 필요

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.model_manager import ModelManager
from src.data_loader.preprocessor import DocumentPreprocessor
from src.data_loader.indexer import GlobalIndexer
from src.core.orchestrator import LongContextOrchestrator
from src.utils.logger import ExperimentLogger  # 분리한 로거 모듈

def run_experiment(config, logger): # logger 인자 추가
    """
    하나의 실험 세트(파일들 + 질문들)를 실행합니다.
    """
    print(f"\n{'='*20} Starting Experiment: {config['name']} {'='*20}")
    
    # 1. 초기화
    model_mgr = ModelManager()
    model_mgr.load_model()
    
    preprocessor = DocumentPreprocessor(max_tokens_per_chunk=1000)
    indexer = GlobalIndexer()
    orchestrator = LongContextOrchestrator(model_mgr)

    # 2. 모든 소스 파일 로드 및 통합
    full_text = ""
    for file_path in config['source_files']:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            full_text += f.read() + "\n\n"

    if not full_text.strip():
        print("Error: No content to process.")
        return

    chunks = preprocessor.chunk_text(full_text, config['name'])
    print(f"Total processed chunks: {len(chunks)}")
    structural_map = indexer.build_structural_map(chunks)

    # 3. 쿼리 반복 실행
    for query in config['queries']:
        print(f"\n[Running Query] {query}")
        
        # 지연 시간 측정을 위해 시작 시간 기록
        start_time = time.time()
        
        answer = orchestrator.process_query(query, chunks, structural_map)
        
        # 로거를 통한 결과 기록
        logger.log_result(
            exp_name=config['name'],
            query=query,
            answer=answer,
            start_time=start_time,
            files=config['source_files']
        )
        
        print("\n" + "-"*30)
        print(f"[Result for: {config['name']}]")
        print(answer)
        print("-" * 30)
        print(f"✅ Result logged to CSV.")

def main(config_path : str):
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    # 실험 전체를 관리할 로거 인스턴스 생성
    logger = ExperimentLogger()

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 모든 실험 세트 순회
    for exp_config in config_data.get('experiments', []):
        run_experiment(exp_config, logger) # logger 전달

if __name__ == "__main__":
    main(config_path="configs/experiment_config_txt.yaml")