import os
import csv
import time
from datetime import datetime
from typing import Dict, Any

class ExperimentLogger:
    """
    실험 결과를 CSV 파일로 통합 관리하는 로거 모듈입니다.
    """
    def __init__(self, base_path: str = "results"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self.filename = os.path.join(self.base_path, "experiment_log.csv")
        self.headers = ["timestamp", "experiment_name", "query", "answer", "latency", "source_files"]
        self._init_csv()

    def _init_csv(self):
        """파일이 없으면 헤더를 생성합니다."""
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_result(self, exp_name: str, query: str, answer: str, start_time: float, files: list):
        """
        한 건의 실험 결과를 파일에 기록합니다.
        """
        latency = round(time.time() - start_time, 2)
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": exp_name,
            "query": query,
            "answer": answer.replace("\n", " "),  # CSV 포맷 유지
            "latency": f"{latency}s",
            "source_files": ", ".join(files)
        }

        with open(self.filename, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(data)