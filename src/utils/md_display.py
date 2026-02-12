import os
from typing import Any
from IPython.display import display, Markdown, HTML

def display_formatted_response(text: str):
    """
    LLM이 생성한 마크다운 텍스트를 사람이 보기 좋은 형태(HTML/Markdown)로 렌더링합니다.
    - 주피터 노트북(experiment.ipynb) 환경에서 최적화되어 작동합니다.
    - 테이블의 테두리와 배경색을 추가하여 가독성을 높입니다.
    """
    # 1. 텍스트 내의 과도한 공백 제거 및 줄바꿈 정제
    cleaned_text = text.strip()
    
    # 2. Markdown 렌더링
    # 기본 Markdown display는 가끔 표의 스타일이 밋밋하므로 CSS를 살짝 가미합니다.
    style = """
    <style>
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-family: sans-serif; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f4f4f4; color: #333; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        pre { background-color: #272822; color: #f8f8f2; padding: 10px; border-radius: 5px; }
    </style>
    """
    
    # HTML과 Markdown을 동시에 출력 (CSS + Markdown 내용)
    display(HTML(style))
    display(Markdown(cleaned_text))

def print_step_header(step_num: int, title: str):
    """단계별 진행상황을 강조하여 출력합니다."""
    header = f"### [STEP {step_num}] {title}"
    display(Markdown("---"))
    display(Markdown(header))