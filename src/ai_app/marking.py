from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# APIRouter 인스턴스 생성
# prefix를 "/ai"로 지정하여 /ai/marking 경로가 되도록 합니다.
router = APIRouter(
    prefix="/ai",
    tags=["Marking Service"]
)

# --- Pydantic 모델 정의 ---
# 1. AI에게 전달할 사용자 답변 데이터 구조
class QuestionInput(BaseModel):
    문제: str = Field(..., description="문제 내용")
    답: str = Field(..., description="문제에 대한 사용자의 답")

# 2. AI에게 요청할 최종 채점 결과 데이터 구조
class MarkedQuestionOutput(BaseModel):
    문제: str = Field(..., description="문제 내용")
    답: str = Field(..., description="문제에 대한 ai의 답")
    정답여부: Literal["true", "false"] = Field(..., description="문제에 대한 사용자의 답이 맞으면 true, 틀리면 false")

# 3. FastAPI 요청 본문(Request Body) 모델
class MarkingRequest(BaseModel):
    questions: List[QuestionInput] = Field(..., description="채점에 필요한 문제와 사용자의 답 리스트")

# 4. FastAPI 응답 본문(Response Body) 모델 (필드명 수정 및 오류 수정)
class MarkingResponse(BaseModel):
    marked_questions: List[MarkedQuestionOutput]
    # 이전 오류: Field()는 위치 인자를 하나만 받음. 수정: description= 사용
    correct_num: int = Field(..., description="총 맞은 문제의 개수")
    incorrect_num: int = Field(..., description="총 틀린 문제의 개수")
    score: int = Field(..., description="총 점수")


# --- LLM 및 LCEL 체인 설정 ---
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1", 
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.0
)

SYSTEM_PROMPT = """
당신은 전문 채점관입니다. 주어진 문제를 풀고, 사용자의 '답' 필드와 비교하여 정답 여부를 판단해야 합니다. 
문제의 정답을 추론하여 사용자의 답변과 엄격하게 비교하세요.
수학 문제의 경우, 계산 결과가 동일하면 정답으로 간주합니다.
단답형/주관식 문제의 경우, 의미가 동일하거나 오타가 경미하면 정답으로 간주하세요.
OX 퀴즈 문제의 경우, 사용자의 답이 true이면 O, false이면 X를 의미합니다.
결과는 반드시 요청된 JSON 스키마 형식으로만 응답해야 합니다.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "아래 [사용자 답변 데이터]를 보고 채점하여 요청된 JSON 형식으로만 반환하세요:\n\n[사용자 답변 데이터]\n{input_json_string}")
])

# AI가 MarkingResponse 객체를 반환하도록 요청 (MarkedQuestionOutput의 리스트를 포함)
structured_llm = llm.with_structured_output(MarkingResponse)

marking_chain = (
    RunnablePassthrough.assign(
        input_json_string=lambda x: json.dumps([q.dict() for q in x["request_data"].questions], ensure_ascii=False, indent=2)
    )
    | prompt
    | structured_llm
)

# --- 채점 API 엔드포인트 ---
@router.post("/marking", response_model=MarkingResponse) # APIRouter를 사용하므로 @router.post로 변경
async def marking(
    request_data: MarkingRequest = Body(..., description="채점 요청 데이터")
):
    """
    사용자의 문제-답변 쌍을 받아 AI를 통해 채점하고, 
    정답 여부, 맞은 개수, 틀린 개수, 총 점수를 반환합니다.
    """
    if not request_data.questions:
        raise HTTPException(status_code=400, detail="채점할 문제 목록이 비어 있습니다.")
    try:
        # LangChain Chain 실행 -> MarkingResponse 객체 전체를 반환
        result: MarkingResponse = await marking_chain.ainvoke({"request_data": request_data})

        return result
        
    except Exception as e:
        print(f"채점 중 오류 발생: {e}")
        # 실제 환경에서는 AI 응답의 JSON 구조 오류를 처리해야 합니다.
        raise HTTPException(status_code=500, detail=f"AI 채점 서비스 오류: {str(e)}")


    
# --- 실행 예제 (터미널에서 FastAPI 실행 후 테스트 가능) ---
# Uvicorn 실행: uvicorn fastapi_marking_api:app --reload

# curl 예제 (JSON 데이터 전송):
"""
curl -X POST "http://127.0.0.1:8000/ai/marking" \
-H "Content-Type: application/json" \
-d '{
  "questions": [
    { "문제": "1. 1+2=?", "답": "3" },
    { "문제": "2. 2+2=?", "답": "4" },
    { "문제": "3. 한국의 수도는?", "답": "오타와" },
    { "문제": "4. 3의 제곱근은 무엇인가요?", "답": "9" }
  ]
}'
"""