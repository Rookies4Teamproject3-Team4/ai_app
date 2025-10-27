# rag_pipeline.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os, re, json
from dotenv import load_dotenv

# -------------------- ENV 로드 --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path)

# -------------------- 모델 스키마 --------------------
class QAItem(BaseModel):
    question: str = Field(description="문항 본문")
    answer: str = Field(description="문항 정답(서술/객관식/OX 등 텍스트)")
    qtype: Literal["MULTIPLE", "TRUEFALSE", "ESSAY"] = Field(
        description="문제 유형: MULTIPLE(객관식), OX, ESSAY(서술/단답)"
    )
    choices: Optional[List[str]] = Field(
        default=None, description="객관식 보기 리스트(1번부터 순서대로; MULTIPLE일 때만 존재)"
    )

class GenerateQAResponse(BaseModel):
    items: List[QAItem] = Field(description="문제/정답 리스트")

# -------------------- 파서/프롬프트 --------------------
parser = PydanticOutputParser(pydantic_object=GenerateQAResponse)

TEMPLATE = r"""
당신은 학습문제 출제기입니다. 주어진 컨텍스트만 근거로 문제를 생성하세요.
형식 지시를 엄격히 따르고, 다른 텍스트(설명/코드블록/마크다운)는 출력하지 마세요.

요청 옵션:
- 과목: {subject}
- 제목: {title}
- 문제 수: {num_questions}
- 객관식 보기 수(choice_count): {choice_count}
- 서술형 포함(isDesc): {isDesc}
- OX 포함(isOx): {isOx}

규칙:
1) items 배열 길이는 정확히 {num_questions}개.
2) isOx=false면 TRUEFALSE 유형 문항 생성 절대 금지.
3) isDesc=false면 ESSAY(서술/단답) 문항 생성 절대 금지.
4) isDesc=true면 ESSAY(서술/단답)유형 문항을 최소 1문제 이상 생성.
5) isOx=true면 TRUEFALSE 유형 문항을 최소 1문제 이상 생성.

6) MULTIPLE(객관식) 문항을 만들 때는:
   - choices 배열을 반드시 포함하고 길이는 choice_count와 같아야 한다.
   - answer는 choices 중 하나여야 하며 선지 번호로 답하지 말고 선지의 내용으로 답한다. 
7) TRUEFALSE(OX퀴즈) 문항을 만들 때는:
   - answer는 true 또는 false로 답한다.
8) 모든 문항/정답은 컨텍스트에 근거해야 하며 환각 금지.
9) 출력 문자열에서 역슬래시(\\)는 JSON 규격에 맞게 반드시 두 번(\\\\)으로 이스케이프하라.
   - 수식 표기가 필요하면 LaTeX 대신 평문을 사용하라. 예) '\\\\vec(a)' 대신 'vec(a)'.
10) 모든 출력은 한국어로 하되, qtype 값은 'MULTIPLE' | 'TRUEFALSE' | 'ESSAY' 중 하나로 고정한다.

컨텍스트(중요도 순):
{context}

{format_instructions}
- items[*].question: 문자열
- items[*].answer: 문자열
- items[*].qtype: 'MULTIPLE' | 'TRUEFALSE' | 'ESSAY'
- items[*].choices: MULTIPLE일 때만 존재하며 문자열 리스트(길이=choice_count)
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE).partial(
    format_instructions=parser.get_format_instructions()
)

# -------------------- LLM/체인 --------------------
def _get_model() -> ChatOpenAI:
    
    api_key =os.getenv("OPENAI_API_KEY")
    model_id ="gpt-4o"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다 (.env 설정).")
    return ChatOpenAI(api_key=api_key, model=model_id, temperature=0.2)

_model = None
_chain_raw = None 

def get_model():
    global _model
    if _model is None:
        _model = _get_model()
    return _model

def get_chain_raw():
    """LLM 원문 문자열을 먼저 받는 체인 (이후 안전 파싱)."""
    global _chain_raw
    if _chain_raw is None:
        _chain_raw = prompt | get_model() | StrOutputParser()
    return _chain_raw

# -------------------- 안전 파서 유틸 --------------------
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _strip_json_fences(s: str) -> str:
    return _JSON_FENCE_RE.sub("", s).strip()

def _escape_invalid_backslashes(s: str) -> str:
    r"""
    JSON에서 유효하지 않은 \x 이스케이프를 \\\\x로 보정.
    (\", \\, \/, \b, \f, \n, \r, \t, \u만 허용)
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

def _safe_parse_to_model(text: str) -> GenerateQAResponse:
    raw = _strip_json_fences(text)
    try:
        return parser.parse(raw)
    except Exception:
        
        fixed = _escape_invalid_backslashes(raw)
        try:
            return parser.parse(fixed)
        except Exception:
            
            data = json.loads(fixed)
            return GenerateQAResponse.model_validate(data)

# -------------------- MOCK 생성기(선택) --------------------
def _mock_items(num_questions: int, choice_count: Optional[int]) -> GenerateQAResponse:
    cc = choice_count or 4
    items: List[QAItem] = []
    for i in range(1, num_questions + 1):
        qtype: Literal["MULTIPLE", "TRUEFALSE", "ESSAY"] = "MULTIPLE"
        choices = [f"보기 {j}" for j in range(1, cc + 1)] if qtype == "MULTIPLE" else None
        items.append(QAItem(
            question=f"더미 문항 {i}: 컨텍스트의 핵심 개념을 고르시오.",
            answer=choices[0] if choices else "정답",
            qtype=qtype,
            choices=choices
        ))
    return GenerateQAResponse(items=items)

# -------------------- 공개 함수 --------------------
def generate_qa_with_parser(
    *,
    subject: str,
    title: str,
    num_questions: int,
    choice_count: Optional[int],
    isDesc: bool,
    isOx: bool,
    context: str
) -> GenerateQAResponse:
    
    if os.getenv("MOCK_AI", "").lower() in ("1", "true", "yes", "on"):
        return _mock_items(num_questions, choice_count)

    text = get_chain_raw().invoke({
        "subject": subject,
        "title": title,
        "num_questions": num_questions,
        "choice_count": choice_count if choice_count is not None else "N/A",
        "isDesc": isDesc,
        "isOx": isOx,
        "context": context,
    })
    return _safe_parse_to_model(text)

def to_numbered_key_list(resp: GenerateQAResponse) -> list[dict]:
    """
    백엔드 저장형식으로 변환:
    - question: MULTIPLE이면 "문제문  1번 a  2번 b ..." 형태로 보기 포함
    - type: 'MULTIPLE' | 'TRUEFALSE' | 'ESSAY'
    - answer: 정답 텍스트, 문제 유형이 TRUEFALSE일 경우 답은 true 또는 false로 대답
    """
    out: List[dict] = []
    for idx, item in enumerate(resp.items, start=1):
        display_q = item.question.strip()
        if item.qtype == "MULTIPLE" and item.choices:
            joined = "  ".join(f"{i}번 {ch}" for i, ch in enumerate(item.choices, start=1))
            display_q = f"{display_q}  {joined}"
        out.append({
            f"question": display_q,
            "type": item.qtype,
            f"answer": item.answer,  
        })
    return out

def answer_question(**kwargs):
    return generate_qa_with_parser(**kwargs)
