from fastapi import APIRouter, Body, HTTPException, Form
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json
import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

# --- build_index.py에서 FAISS 유틸리티를 위한 임시 함수 정의 ---
# 실제 프로젝트에서는 build_index.py의 get_embeddings, load_faiss를 import 해야 합니다.
# ----------------------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
FAISS_DIR = ".faiss"       
def get_embeddings() -> Embeddings:
    # 'text-embedding-3-small' 임베딩 모델을 반환합니다.
    return OpenAIEmbeddings(model=EMBED_MODEL)

def load_faiss(subject: str, embeddings: Embeddings) -> Optional[FAISS]:
    """FAISS 경로 생성 및 로드 (build_index.py의 로직을 간소화하여 재현)"""
    def _sanitize_dir(name: str) -> str:
        import hashlib
        base = re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('-.')
        if not base: base = "subject"
        suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        return f"{base}-{suffix}"

    safe_subject = _sanitize_dir(subject)
    path = os.path.join(FAISS_DIR, safe_subject)

    if os.path.isdir(path):
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"FAISS 로드 오류 (marking): {e}")
            return None
    return None
# ----------------------------------------------------------------------------------

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# APIRouter 인스턴스 생성
router = APIRouter(
   prefix="/ai",
   tags=["Marking Service"]
)

# --- Pydantic 모델 정의 ---
class QuestionInput(BaseModel):
   문제: str = Field(..., description="문제 내용")
   답: str = Field(..., description="문제에 대한 사용자의 답")

class MarkedQuestionOutput(BaseModel):
   문제: str = Field(..., description="문제 내용")
   답: str = Field(..., description="AI가 추론한 정답 텍스트")
   정답여부: Literal["true", "false"] = Field(..., description="사용자의 답이 정답이면 true, 틀리면 false")

class MarkingRequest(BaseModel):
   questions: List[QuestionInput] = Field(..., description="채점에 필요한 문제와 사용자의 답 리스트")
   subject: str = Field(..., description="채점할 문제의 컨텍스트(과목명)")

class MarkingResponse(BaseModel):
   marked_questions: List[MarkedQuestionOutput]
   correct_num: int = Field(..., description="총 맞은 문제의 개수")
   incorrect_num: int = Field(..., description="총 틀린 문제의 개수")
   score: int = Field(..., description="총 점수")
   ai_comment: str = Field(..., description="AI 학습 평가 코멘트 (예: 추가 학습 필요, 학습 완료 등)")


# --- LLM 및 LCEL 체인 설정 ---
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)

SYSTEM_PROMPT = """
당신은 전문 채점관입니다. 아래 [문제 데이터]와 [참조 컨텍스트]를 보고, 각 문제의 정답을 추론하여 사용자의 '답' 필드와 엄격하게 비교하세요.
참조 컨텍스트가 주어지면, 그 컨텍스트를 기반으로 문제를 해결하고 정답을 판단하세요. 컨텍스트가 없다면 일반 지식으로 채점하세요.

규칙:
1. 문제의 정답을 추론하여 사용자의 답변과 비교 후 '정답여부' 필드를 'true' 또는 'false'로 채우세요.
2. '답' 필드는 AI가 추론한 '정답'을 텍스트로 채우세요.
3. 객관식, 주관식, OX 문제 채점 기준은 아래를 따르세요.
  - 객관식: 
    1) **사용자 답변이 추론된 정답 선지 내용과 완전히 동일하거나** 정답 선지 내용이 사용자 답변을 포함하는 경우 정답.
    2) **사용자 답변이 '1', '2', '3', '4' 등의 선지 번호일 경우,** 해당 번호가 **정답 선지의 번호와 일치하면** 정답으로 처리합니다.
  - 단답형/주관식: 의미가 동일하거나 오타가 경미하면 정답.
  - OX 퀴즈: 'true'/'false'를 O/X로 간주합니다.
4. 전체 정답률을 바탕으로 학습 상태를 평가하여 ai_comment를 생성하세요.
5. ai_comment는 아래 예시 중 하나로 구성합니다:
   - 정답률 90% 이상 → "학습 완료, 매우 우수합니다!"
   - 정답률 70~89% → "학습이 잘 진행되고 있습니다. 복습을 권장합니다."
   - 정답률 50~69% → "핵심 개념에 대한 추가 학습이 필요합니다."
   - 정답률 50% 미만 → "기초 개념부터 다시 복습해보세요."
6. 결과는 반드시 요청된 JSON 스키마 형식으로만 응답해야 하고, 'ai_comment' 필드를 포함해야 합니다.

[참조 컨텍스트]
{context}
"""

prompt = ChatPromptTemplate.from_messages([
   ("system", SYSTEM_PROMPT),
   ("human", "아래 [문제 데이터]를 보고 채점하여 요청된 JSON 형식으로만 반환하세요:\n\n[문제 데이터]\n{input_json_string}")
])

# AI가 MarkingResponse 객체를 반환하도록 요청
structured_llm = llm.with_structured_output(MarkingResponse)

# 🚨 [추가] 프롬프트와 LLM을 결합하는 체인을 정의합니다.
marking_chain = prompt | structured_llm




# --- 채점 API 엔드포인트 ---
@router.post("/marking", response_model=MarkingResponse)
async def marking(
# 🚨 [수정] subject 필드를 제거하고 MarkingRequest만 Body로 받음
   request_data: MarkingRequest = Body(..., description="채점 요청 데이터")
):
   # ...
   # 1. 벡터 검색 (RAG)
   # request_data.subject로 접근하도록 변경
   subject = request_data.subject
   try:
      
      embeddings = get_embeddings()
      vectordb = load_faiss(subject, embeddings)
      
      # 1. 벡터 검색 (RAG)
      context_text = ""
      if vectordb:
         # 모든 문제 본문을 합쳐서 검색 쿼리로 사용
         all_questions = " ".join([q.문제 for q in request_data.questions])
         # 더 많은 문서를 검색하여 채점의 정확도를 높입니다.
         docs = vectordb.similarity_search(all_questions, k=8) 
         context_text = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
      else:
         # print를 사용하여 서버 콘솔에 경고 출력
         print(f"경고: Subject '{subject}'에 대한 FAISS 인덱스를 찾을 수 없습니다. 일반 지식으로 채점합니다.")

      # 2. LLM 체인 실행
      # 문제 데이터는 JSON 문자열로 변환
      input_json_string = json.dumps([q.dict() for q in request_data.questions], ensure_ascii=False, indent=2)

      result: MarkingResponse = await marking_chain.ainvoke(
      {
        "input_json_string": input_json_string,
        "context": context_text
      }
    )

      # 3. 맞은 개수, 틀린 개수 계산 (LLM 응답에 의존)
      correct_num = sum(1 for q in result.marked_questions if q.정답여부 == "true")
      incorrect_num = len(result.marked_questions) - correct_num
      total_problems = len(result.marked_questions)
      if total_problems > 0:
         total_score = int(((correct_num / total_problems) * 100.0) + 0.5)


      if total_problems > 0:
         total_score = int(((correct_num / total_problems) * 100.0) + 0.5)

      if total_score >= 90:
         ai_comment = "학습 완료, 매우 우수합니다!"
      elif total_score >= 70:
         ai_comment = "학습이 잘 진행되고 있습니다. 복습을 권장합니다."
      elif total_score >= 50:
         ai_comment = "핵심 개념에 대한 추가 학습이 필요합니다."
      else:
         ai_comment = "기초 개념부터 다시 복습해보세요."


      # 4. 결과 객체에 계산된 값 대입 후 반환
      return MarkingResponse(
         marked_questions=result.marked_questions,
         correct_num=correct_num,
         incorrect_num=incorrect_num,
         score=total_score,
         ai_comment=ai_comment
      )
      
   except Exception as e:
      print(f"채점 중 오류 발생: {e}")
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