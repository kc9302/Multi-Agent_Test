# Multi-Agent Test

이 저장소는 다중 에이전트를 이용한 간단한 학습용 예제 프로젝트입니다. 수학, 영어, 한국사 에이전트와 RAG(Retrieval Augmented Generation)를 결합하여 질문에 답변합니다.

## 빠른 시작

아래 단계에 따라 가상환경을 만들고 필요한 패키지를 설치하면 어떤 환경에서도 바로 실행할 수 있습니다.

```bash
# 1) 파이썬 3.10+ 버전 권장
python3 -m venv venv
source venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

```python
from make_langgraph import activate_agent

question = [{"role": "user", "content": "구석기 시대에 대해 알려줘"}]
result = activate_agent(question)
print(result)
```

`activate_agent` 함수는 에이전트 그래프를 실행하여 결과를 반환합니다.

## 파일 구성

- `multi_agent.py` – 각 에이전트 로직과 RAG 구성이 포함되어 있습니다.
- `make_langgraph.py` – LangGraph 를 사용해 에이전트를 연결하고 `activate_agent` 함수를 제공합니다.
- `finemodel/` – 실험용 모델 가중치와 토크나이저 파일들이 위치합니다.
- `rag_file.txt` – RAG 예시에 사용되는 문서 샘플입니다.

## 라이선스

MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참고하세요.