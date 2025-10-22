# tests/test_rag_pipeline.py
import io
from typing import List

# ---------- 유닛 테스트 (어댑터) ----------
def test_adapter_numbered_keys():
    from ai_app.rag_pipeline import GenerateQAResponse, QAItem, to_numbered_key_list
    resp = GenerateQAResponse(items=[
        QAItem(question="Q1?", answer="A1"),
        QAItem(question="Q2?", answer="A2"),
        QAItem(question="Q3?", answer="A3"),
    ])
    numbered = to_numbered_key_list(resp)
    assert isinstance(numbered, list)
    assert numbered[0]["question1"] == "Q1?"
    assert numbered[0]["answer1"] == "A1"
    assert numbered[2]["question3"] == "Q3?"
    assert numbered[2]["answer3"] == "A3"

# ---------- 유닛 테스트 (MOCK_AI 경로) ----------
def test_generate_qa_with_parser_mock_mode():
    from ai_app.rag_pipeline import generate_qa_with_parser
    resp = generate_qa_with_parser(
        subject="확통1",
        title="확률의기초_1회",
        num_questions=5,
        choice_count=5,
        isDesc=True,
        isOx=False,
        context="표본공간/사건/조건부확률"
    )
    assert len(resp.items) == 5
    assert all(item.question and item.answer for item in resp.items)

# ---------- 엔드포인트 테스트 (/ai/generate-qa) ----------
def test_generate_qa_endpoint_returns_numbered_list(client):
    # 가짜 PDF 바이트 (실제 파싱은 목킹되어 사용되지 않음)
    fake_pdf = io.BytesIO(b"%PDF-1.4\n%fake\n%%EOF")
    files = {
        "pdf": ("dummy.pdf", fake_pdf, "application/pdf")
    }
    data = {
        "subject": "확통1",
        "title": "확률의기초_1회",
        "num_questions": "4",
        "choice_count": "5",
        "isDesc": "true",
        "isOx": "false",
    }
    r = client.post("/ai/generate-qa", files=files, data=data)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) == 4
    
    # 번호 키 검사
    assert "question1" in payload[0] and "answer1" in payload[0]
    assert all(f"question{i+1}" in payload[i] and f"answer{i+1}" in payload[i] for i in range(4))
