# src/ai_app/tests/run_manual_test.py
import os
from pathlib import Path
from pprint import pprint

# 테스트는 src/ai_app 내부 모듈을 직접 임포트
from ai_app.build_index import context_from_pdf_bytes
from ai_app.rag_pipeline import generate_qa_with_parser, to_numbered_key_list

def banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def run_local_pipeline_test(
    pdf_path: str,
    subject: str = "확통1",
    title: str = "확률의기초_1회",
    num_questions: int = 5,
    choice_count: int | None = 5,
    isDesc: bool = True,
    isOx: bool = False,
):
    # 0) MOCK 모드 강제(키 없어도 엔드투엔드 확인)
    os.environ["MOCK_AI"] = "false"

    banner("1) PDF 로드")
    pdf_file = Path(pdf_path)
    assert pdf_file.exists(), f"PDF not found: {pdf_file}"
    pdf_bytes = pdf_file.read_bytes()
    print(f"- 파일: {pdf_file.name} ({len(pdf_bytes)} bytes)")

    banner("2) 인덱싱 → 컨텍스트 생성 (FAISS)")
    context = context_from_pdf_bytes(pdf_bytes, subject=subject, title=title)
    print(context[:600] + ("..." if len(context) > 600 else ""))

    banner("3) 문제 생성 (generate_qa_with_parser)")
    resp = generate_qa_with_parser(
        subject=subject,
        title=title,
        num_questions=num_questions,
        choice_count=choice_count,
        isDesc=isDesc,
        isOx=isOx,
        context=context,
    )
    print(f"- 생성 문항 수: {len(resp.items)}")

    QTYPE_KR = {"MCQ": "객관식", "OX": "OX", "ESSAY": "서술형"}

    for i, item in enumerate(resp.items, start=1):
        qtype_kr = QTYPE_KR.get(getattr(item, "qtype", ""), "")
        header = f"[{i}] ({qtype_kr}) Q: {item.question}" if qtype_kr else f"[{i}] Q: {item.question}"
        print("\n" + header)

        choices = getattr(item, "choices", None)
        if getattr(item, "qtype", "") == "MCQ" and choices:
            for j, ch in enumerate(choices, start=1):
                print(f"    {j}) {ch}")

        print(f"    A: {item.answer}")


    banner("4) 번호 키 형태(백엔드 저장 형식)")
    numbered = to_numbered_key_list(resp)
    pprint(numbered)


if __name__ == "__main__":
    pdf = "src/ai_app/data/comparison_test.pdf"
    run_local_pipeline_test(
        pdf_path=pdf,
        subject="수열",
        title="수열_1회",
        num_questions=5,
        choice_count=4,
        isDesc=False,
        isOx=False,
    )
