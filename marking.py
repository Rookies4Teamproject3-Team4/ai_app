from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
import json
import time
import httpx # 비동기 HTTP 요청을 위한 라이브러리
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
# .env 파일에서 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "") # 실제 사용 시 환경 변수 설정 필요

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

# 4. FastAPI 응답 본문(Response Body) 모델
class MarkingResponse(BaseModel):
    marked_questions: List[MarkedQuestionOutput]
    correct_num: int = Field(..., description="총 맞은 문제의 개수")
    incorrect_num: int = Field(..., description="총 틀린 문제의 개수")
    score: int = Field(..., description="총 점수")
# FastAPI 앱 초기화
app = FastAPI(title="AI Marking Service")

# .env 파일을 불러와서 환경 변수로 설정
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY[:5])

# Groq 또는 Moonshot 등 OpenAI 호환 API 엔드포인트를 사용하도록 ChatOpenAI 설정
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1", 
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.0 # 채점은 정확해야 하므로 낮은 온도로 설정
)

# 1. 시스템 프롬프트 정의
SYSTEM_PROMPT = """
당신은 전문 채점관입니다. 주어진 문제를 풀고, 사용자의 '답' 필드와 비교하여 정답 여부를 판단해야 합니다. 
문제의 정답을 추론하여 사용자의 답변과 엄격하게 비교하세요.
수학 문제의 경우, 계산 결과가 동일하면 정답으로 간주합니다.
단답형/주관식 문제의 경우, 의미가 동일하거나 오타가 경미하면 정답으로 간주하세요.
OX 퀴즈 문제의 경우, 사용자의 답이 true이면 O, false이면 X를 의미합니다.
결과는 반드시 요청된 JSON 스키마 형식으로만 응답해야 합니다.
"""

# 2. ChatPromptTemplate 정의 (LCEL 사용)
# 사용자 입력은 JSON 문자열로 받을 예정이므로, {input_json_string} 변수를 사용합니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "아래 [사용자 답변 데이터]를 보고 채점하여 요청된 JSON 형식으로만 반환하세요:\n\n[사용자 답변 데이터]\n{input_json_string}")
])

# 3. LLM을 Structured Output Chain으로 변환
# MarkingResponse 모델 스키마를 사용하여 LLM이 JSON을 출력하도록 강제합니다.
structured_llm = llm.with_structured_output(MarkingResponse)

# 4. 최종 LCEL Chain 구성
# RunnablePassthrough.assign을 사용하여 FastAPI로 받은 request_data를 JSON 문자열로 변환하여 프롬프트에 전달합니다.
marking_chain = (
    RunnablePassthrough.assign(
        # MarkingRequest 객체를 JSON 문자열로 변환하여 'input_json_string' 변수에 할당
        input_json_string=lambda x: json.dumps([q.dict() for q in x["request_data"].questions], ensure_ascii=False, indent=2)
    )
    | prompt # {input_json_string}을 포함한 프롬프트 적용
    | structured_llm # 구조화된 응답 요청
)

# --- FastAPI 엔드포인트 수정 ---

import asyncio # asyncio.sleep 사용을 위해 추가

@app.post("/ai/marking", response_model=MarkingResponse, tags=["AI Services"])
async def marking(
    # Body를 사용하여 JSON 본문을 Pydantic 모델로 직접 받습니다.
    # 기존 Form 파라미터(subject, title 등)는 채점 로직과 직접 관련 없으므로 제거했습니다.
    # 필요하다면 @app.post("/ai/marking/{subject}/{title}") 처럼 경로 파라미터로 받거나,
    # MarkingRequest 모델 내부에 포함시킬 수 있습니다.
    request_data: MarkingRequest = Body(..., description="List of questions and user answers to mark")
):
    """
    사용자의 문제-답변 쌍을 받아 AI(Gemini)를 통해 채점하고, 
    정답 여부가 포함된 JSON 목록을 반환합니다.
    """
    
    if not request_data.questions:
        raise HTTPException(status_code=400, detail="채점할 문제 목록이 비어 있습니다.")
    try:
        # LangChain Chain을 비동기적으로 실행합니다.
        # Chain의 입력은 {"request_data": MarkingRequest 객체} 형태입니다.
        result: MarkingResponse = await marking_chain.ainvoke({"request_data": request_data})

        # LangChain structured_llm의 결과는 MarkingResponse 객체이므로 바로 반환합니다.
        return result
    except Exception as e:
        # API 오류 또는 JSON 파싱 오류 처리
        print(f"채점 중 오류 발생: {e}")
        # 실제 환경에서는 로깅을 상세하게 남겨야 합니다.
        raise HTTPException(status_code=500, detail=f"AI 채점 서비스 오류: {str(e)}")




    # 1. LLM에게 전달할 프롬프트 구성
    
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