# 상품 리뷰 감성 분석기

쇼핑몰 리뷰 텍스트를 붙여넣으면 Gemini 기반 에이전트가 **장점/단점 분류 → 감성 점수(1~5점) 산정 → 실제 별점과 비교한 과장 리뷰 리포트**를 만들어 줍니다.

> 수행평가 4 대체 과제 「나를 위한 AI 직원 만들기」 — 주제 ③ 상품 리뷰 감성 분석기

---

## 무엇을 하나요

리뷰에 별 5개가 달려 있어도, 막상 읽어보면 "색상이 다르고 마감도 거칠다"는 불만이 가득한 경우가 있습니다.
이 도구는 **별점이 아니라 리뷰 내용만 보고** 점수를 매긴 뒤, 실제 별점과 비교해서 그 차이를 알려줍니다.

| 입력 | 출력 |
|---|---|
| 리뷰 원문 텍스트 (5~2000자) | 장점 목록 / 단점 목록 |
| (선택) 실제 별점 1~5 | 감성 점수 1~5점 + 근거 |
| | 과장 리뷰 여부 리포트 |

---

## 에이전트가 쓰는 도구 3개

Gemini의 **Function Calling** 으로 아래 3개 함수를 등록해 두고, 모델이 상황에 맞게 직접 골라 호출합니다.

| 도구 | 하는 일 | 입력 | 출력 | LLM 호출 |
|---|---|---|---|---|
| `classify_review_text` | 리뷰를 장점/단점 문장으로 분류 | 리뷰 원문 | `pros[]`, `cons[]` | O |
| `score_sentiment` | 리뷰 내용만 보고 별점 추정 | 리뷰 원문 | `score`, `reason` | O |
| `detect_review_exaggeration` | 별점과 감성 점수의 괴리 계산 | 실제 별점, 감성 점수 | `gap`, `verdict`, `explanation` | **X** |

3번 도구는 의도적으로 **LLM을 쓰지 않습니다.** 단순 뺄셈과 비교로 충분한 일에 API 호출을 쓰면 무료 등급 한도만 소모되기 때문입니다. 덕분에 API 키 없이도 테스트할 수 있습니다.

### 과장 판단 기준

```
gap = 실제 별점 − 감성 점수

gap ≥ +1.5  →  과장 의심      (내용보다 별점이 후하다)
gap ≤ −1.5  →  과소평가 의심  (내용보다 별점이 박하다)
그 사이      →  대체로 일치
```

> 기획서에는 "두 점수 차이가 임계값 이상이면 과장"이라고만 적혀 있고 **차이의 방향**이 명시되어 있지 않았습니다.
> "과장"은 별점이 내용보다 후한 경우를 뜻하므로 양수 방향으로 한정하고, 반대 방향(과소평가)은 별도 플래그로 함께 제공하도록 정했습니다.
> 임계값 1.5는 기본값이며 화면 사이드바에서 0.5~3.0 사이로 바꿀 수 있습니다.

---

## 실행 흐름

```
사용자 입력 (리뷰 + 선택적 실제 별점)
        │
        ▼
  Gemini 에이전트  ─── 어떤 도구가 필요한지 스스로 판단
        │
        ├──▶ classify_review_text      → 장점/단점
        ├──▶ score_sentiment           → 감성 점수
        └──▶ detect_review_exaggeration → 괴리 계산 (별점을 입력했을 때만)
        │
        ▼
  결과 종합 → Streamlit 화면에 카드/표로 출력
```

SDK의 **자동 함수 호출 기능을 일부러 끄고** 호출 루프를 직접 돌립니다.
그래야 "에이전트가 어떤 도구를 어떤 인자로 불렀는지"를 기록해서 화면에 보여줄 수 있습니다.

---

## 기술 스택

| 구성 요소 | 선택 | 버전 |
|---|---|---|
| LLM | Google Gemini API (무료 등급) | `gemini-2.0-flash` |
| SDK | `google-genai` | 1.47.0 |
| 도구 호출 | Gemini Function Calling | — |
| UI/서버 | `streamlit` | 1.50.0 |
| 환경변수 | `python-dotenv` | 1.2.1 |
| 테스트 | `pytest` | 8.4.2 |
| Python | CPython | 3.9.13 |

> **SDK 선택에 대해:** 기획서는 `google-generativeai` 를 지정했지만, 이 패키지는 Google이 2025년 8월 지원 종료를 공지한 레거시 SDK입니다.
> 현재 공식 후속인 `google-genai` 로 변경했습니다. 두 SDK는 함수 호출 개념은 같지만 API 표기가 다릅니다.

---

## 설치 및 실행

### 1. 저장소 내려받기

```powershell
git clone https://github.com/neeeeding/Product-Review-Sentiment-Analyzer.git
cd Product-Review-Sentiment-Analyzer
```

### 2. 가상환경 만들고 패키지 설치

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

> 설치 중 `ConnectionResetError` 로 끊기면 재시도 옵션을 붙이세요.
> `venv\Scripts\python.exe -m pip install -r requirements.txt --retries 10 --timeout 120`

### 3. API 키 발급

1. https://aistudio.google.com/ 접속 → 로그인
2. **Get API key → Create API key** 클릭
3. 발급된 키 복사

### 4. API 키 설정

```powershell
Copy-Item .env.example .env
```

`.env` 파일을 열어 실제 키를 붙여넣습니다.

```
GOOGLE_API_KEY=AIzaSy...여기에_실제_키
```

> `.env` 는 `.gitignore` 에 등록되어 있어 **깃허브에 절대 올라가지 않습니다.**
> 키를 코드에 직접 적거나 커밋하지 마세요.

### 5. 연결 확인 (권장)

앱을 켜기 전에 키가 제대로 설정됐는지 먼저 확인합니다.

```powershell
venv\Scripts\python.exe scripts/check_connection.py
```

키 확인 → 사용 가능한 모델 목록 조회 → 설정된 모델 사용 가능 여부를 차례로 점검합니다.
화면에 출력되는 키는 앞 6자와 뒤 4자만 남기고 가려집니다.

### 6. 앱 실행

```powershell
venv\Scripts\streamlit.exe run app.py
```

브라우저에서 http://localhost:8501 이 열립니다.

---

## 코드 구조

```
Product-Review-Sentiment-Analyzer/
├── app.py                      # Streamlit 화면 (입력 폼, 결과 카드, 도구 호출 기록)
├── src/
│   ├── config.py               # .env에서 API 키/모델 로드, 길이 제한 상수
│   ├── gemini_client.py        # 모든 Gemini 호출 + 재시도/백오프 + 오류 한국어화
│   ├── tools.py                # 도구 3개 구현
│   └── agent.py                # 도구 스키마 등록 + function calling 루프
├── scripts/
│   └── check_connection.py     # API 연결 점검 스크립트
├── tests/                      # 테스트 134개 (실제 API 호출 없음)
│   ├── test_tools.py           #   도구③ 계산 로직
│   ├── test_llm_tools.py       #   도구①② 파싱/검증 (가짜 Gemini)
│   ├── test_gemini_client.py   #   재시도/백오프
│   ├── test_agent.py           #   함수 호출 루프 (대본 방식 가짜 Gemini)
│   ├── test_config.py          #   API 키 로딩
│   └── test_app.py             #   Streamlit 화면 (AppTest)
├── docs/
│   └── 개발일지.md
├── requirements.txt
├── .env.example                # 키 입력 템플릿 (실제 .env는 커밋 제외)
└── pytest.ini
```

### 왜 이렇게 나눴나

- **`gemini_client.py` 에 API 호출을 전부 모은 이유** — 재시도 로직을 한 군데만 고치면 되게 하려고요. 도구마다 재시도를 따로 짜면 한 곳을 고칠 때 다른 곳을 빠뜨립니다.
- **`tools.py` 와 `agent.py` 를 나눈 이유** — 도구는 "무엇을 하는가", 에이전트는 "언제 부를 것인가"를 담당합니다. 도구는 에이전트 없이 단독으로도 호출·테스트할 수 있습니다.
- **`config.py` 를 따로 둔 이유** — API 키가 코드 어디에도 하드코딩되지 않도록 읽는 창구를 하나로 제한했습니다.

---

## 테스트

```powershell
venv\Scripts\python.exe -m pytest
```

테스트 134개 전부 **실제 Gemini API를 호출하지 않습니다.** `generate` 함수를 가짜로 바꿔치기해서 돌리기 때문에, API 키가 없어도 무료 등급 한도를 쓰지 않아도 실행됩니다.

```
tests/test_tools.py         31개  도구③ 계산, 경계값, 입력 검증
tests/test_llm_tools.py     30개  도구①② JSON 파싱, 범위 clamp, 길이 제한
tests/test_agent.py         24개  도구 호출 루프, 오류 처리, 무한루프 차단
tests/test_app.py           21개  화면 렌더링, 키 없을 때 안내, 결과 표시
tests/test_gemini_client.py 16개  429/503 재시도, 400/403 즉시 중단, 백오프
tests/test_config.py        12개  키 로딩 우선순위, 템플릿 값 감지
```

---

## 무료 등급 사용 시 알아둘 것

### API 호출 횟수

리뷰 1건을 분석하면 Gemini를 **약 4~5번** 호출합니다.

```
에이전트 1턴 (어떤 도구를 부를지 판단)     1회
  └ classify_review_text 내부 호출          1회
  └ score_sentiment 내부 호출               1회
  └ detect_review_exaggeration              0회 (계산만)
에이전트 최종 리포트 작성                   1~2회
```

연속으로 여러 건을 분석하면 분당 한도에 걸릴 수 있습니다.

### 한도에 걸렸을 때

429 응답을 받으면 자동으로 **2초 → 4초 → 8초** 간격(무작위 지터 포함)으로 최대 4번까지 재시도합니다.
그래도 안 되면 화면에 "잠시 기다렸다 다시 시도해 주세요" 안내가 뜹니다. 보통 1분 뒤에 다시 열립니다.

### 입력 길이

리뷰는 **2000자**까지만 받습니다. 토큰 제한에 걸리는 것을 미리 막기 위한 제한이며, 길이 초과 시 API를 호출하기 전에 화면에서 거부합니다.

---

## 트러블슈팅

| 증상 | 원인과 해결 |
|---|---|
| `Gemini API 키를 찾을 수 없습니다` | `.env` 파일이 없거나 `GOOGLE_API_KEY` 가 비어 있습니다. 3~4단계를 다시 하세요. |
| `.env 파일의 GOOGLE_API_KEY 가 아직 안내 문구 그대로입니다` | `.env.example` 을 복사만 하고 키를 안 바꿨습니다. 실제 키로 교체하세요. |
| `API 키에 권한이 없습니다 (403)` | 키가 비활성화됐거나 잘못 복사됐습니다. Google AI Studio에서 키를 다시 확인하세요. |
| `모델을 찾을 수 없습니다 (404)` | `.env` 의 `GEMINI_MODEL` 이름이 틀렸습니다. `scripts/check_connection.py` 로 사용 가능한 모델 목록을 확인하세요. |
| `요청 한도를 초과했습니다 (429)` | 무료 등급 분당 한도입니다. 1분 정도 기다렸다 다시 시도하세요. |
| pip 설치 중 `ConnectionResetError` | 대용량 패키지(pyarrow 26MB) 다운로드가 끊긴 것입니다. `--retries 10 --timeout 120` 을 붙여 다시 실행하세요. |
| `You are using a Python version 3.9 past its end of life` | Python 3.9가 지원 종료되어 `google-auth` 가 띄우는 경고입니다. **현재 동작에는 문제가 없습니다.** 경고를 없애려면 Python 3.10 이상으로 올리세요. |
| `git: command not found` (Windows) | git이 PATH에 없습니다. Git for Windows를 설치하거나, Visual Studio 번들 git의 전체 경로를 쓰세요. |

---

## 보안 메모

- API 키는 `.env` 에만 저장하고 코드에는 절대 적지 않습니다.
- `.gitignore` 에 `.env`, `*.key`, `.streamlit/secrets.toml` 을 등록해 두었습니다.
- `check_connection.py` 는 키를 출력할 때 앞 6자·뒤 4자만 남기고 가립니다. 화면 공유나 스크린샷으로 키가 새는 것을 막기 위해서입니다.
- 스크린샷을 찍기 전에 `.env` 파일이나 터미널 기록에 키가 노출되지 않았는지 확인하세요.

---

## 개발 현황

- [x] 0단계 — 프로젝트 뼈대, `.gitignore`, `.env.example`
- [x] 1단계 — 가상환경 및 의존성 설치
- [x] 2단계 — 도구③ 과장 탐지 (순수 계산) + 단위 테스트
- [x] 3단계 — Gemini 클라이언트 래퍼 (재시도/backoff)
- [x] 4단계 — 도구① 분류, 도구② 감성 점수
- [x] 5단계 — 에이전트 (function calling 오케스트레이션)
- [x] 6단계 — Streamlit UI
- [ ] 7단계 — 실제 리뷰 테스트 + 스크린샷
- [ ] 8단계 — 사용 후기

개발 과정의 판단 근거와 시행착오는 [docs/개발일지.md](docs/개발일지.md) 에 정리했습니다.
