# tests/conftest.py
import os
import sys
from pathlib import Path
import pytest

# --- src/ 를 파이썬 경로에 추가 (pytest 실행 시에도 ai_app import 가능하게) ---
ROOT = Path(__file__).resolve().parents[1]        
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- 전역 환경: LLM 호출 없이 동작하도록 MOCK_AI 활성화 ---
@pytest.fixture(autouse=True)
def _set_mock_ai_env(monkeypatch):
    monkeypatch.setenv("MOCK_AI", "true")
   
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

# --- FastAPI TestClient 제공 ---
@pytest.fixture()
def client(monkeypatch):
    
    import ai_app.api as api_module
    monkeypatch.setattr(api_module, "context_from_pdf_bytes", lambda *_args, **_kw: "DUMMY CONTEXT")
    from fastapi.testclient import TestClient
    return TestClient(api_module.app)
