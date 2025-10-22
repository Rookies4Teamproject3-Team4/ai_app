# src/ai_app/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from ai_app.rag_pipeline import generate_qa_with_parser, to_numbered_key_list
from ai_app.build_index import context_from_pdf_bytes

# ===== .env 경로 로드 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path)

# ===== FastAPI 앱 생성 =====
app = FastAPI(title="Study Helper AI", version="1.0.0")

@app.post("/ai/generate-qa")
async def generate_qa(
    pdf: UploadFile = File(..., description="학습 PDF"),
    subject: str = Form(..., description="과목명 (예: 확통1)"),
    title: str = Form(..., description="세트 제목 (예: 확률의기초_1회)"),
    num_questions: int = Form(10, description="문항 수"),
    choice_count: int | None = Form(None, description="객관식 보기 수 (4/5 등)"),
    isDesc: bool = Form(True, description="서술형 포함 여부"),
    isOx: bool = Form(False, description="OX 포함 여부")
):
    if pdf.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="PDF 파일을 업로드하세요.")

    pdf_bytes = await pdf.read()

    )
    try:
        context = context_from_pdf_bytes(pdf_bytes, subject=subject, title=title)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"인덱싱 실패: {e}")

    )
    try:
        resp = generate_qa_with_parser(
            subject=subject,
            title=title,
            num_questions=num_questions,
            choice_count=choice_count,
            isDesc=isDesc,
            isOx=isOx,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 생성 실패: {e}")

    
    numbered = to_numbered_key_list(resp)
    return JSONResponse(content=numbered, media_type="application/json")
