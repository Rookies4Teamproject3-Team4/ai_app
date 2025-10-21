# rag_pipeline.py
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

# ===== Pydantic 모델 (정규 구조) =====
class QAItem(BaseModel):
    question: str = Field(description="문항 본문")
    answer: str = Field(description="문항 정답(서술/객관식/TF/OX 등 텍스트)")

class GenerateQAResponse(BaseModel):
    items: List[QAItem] = Field(description="문제/정답 리스트")

# ===== 파서 =====
parser = PydanticOutputParser(pydantic_object=GenerateQAResponse)

# ===== 프롬프트 템플릿 (isDesc, isOx 옵션) =====
TEMPLATE = """
당신은 학습문제 출제기입니다. 주어진 컨텍스트만 근거로 문제를 생성하세요.
형식 지시를 엄격히 따르고, 다른 텍스트(설명/코드블록/마크다운)는 출력하지 마세요.

요청 옵션:
- 과목: {subject}
- 제목: {title}
- 문제 수: {num_questions}
- 객관식 보기 수: {choice_count}
- 서술형 포함(isDesc): {isDesc}
- OX 포함(isOx): {isOx}

규칙:
1) items 배열 길이는 정확히 {num_questions}개.
2) isOx=false면 OX 유형 문항 금지.
3) isDesc=false면 서술형 문항 금지.
4) 모든 문항/정답은 컨텍스트에 근거해야 하며 환각 금지.

컨텍스트(중요도 순):
{context}

{format_instructions}
- items[*].question: 문자열
- items[*].answer: 문자열
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE).partial(
    format_instructions=parser.get_format_instructions()
)

# ===== 모델 =====
def _get_model() -> ChatOpenAI:
    base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1"
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_id = os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
    if not api_key:
        raise RuntimeError("GROQ_API_KEY 또는 OPENAI_API_KEY가 필요합니다 (.env 설정).")
    return ChatOpenAI(base_url=base_url, api_key=api_key, model=model_id, temperature=0.2)

_model = None
def get_model():
    global _model
    if _model is None:
        _model = _get_model()
    return _model

chain = prompt | get_model() | parser

# ===== 체인 호출 함수 =====
def generate_qa_with_parser(
    *,
    subject: str,
    title: str,
    num_questions: int,
    choice_count: Optional[int],
    isDesc: bool,     # 서술형 포함
    isOx: bool,       # OX 포함
    context: str
) -> GenerateQAResponse:
    return chain.invoke({
        "subject": subject,
        "title": title,
        "num_questions": num_questions,
        "choice_count": choice_count if choice_count is not None else "N/A",
        "isDesc": isDesc,
        "isOx": isOx,
        "context": context[:12000],
    })

# ===== 어댑터: 정규 구조 → 번호 키 리스트 =====
def to_numbered_key_list(resp: GenerateQAResponse) -> list[dict]:
    out = []
    for idx, item in enumerate(resp.items, start=1):
        out.append({f"question{idx}": item.question, f"answer{idx}": item.answer})
    return out
