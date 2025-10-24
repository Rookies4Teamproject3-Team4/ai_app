# ---- 베이스 이미지: Ubuntu 기반으로 변경 (PyTorch 호환성) ----
FROM python:3.12-slim AS builder

WORKDIR /app

# 빌드 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# pyproject.toml에 명시된 의존성 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "python-dotenv>=1.1.1,<2.0.0" \
    "langchain>=1.0.1,<2.0.0" \
    "langchain-openai>=1.0.0,<2.0.0" \
    "fastapi>=0.119.1,<0.120.0" \
    "pypdf>=6.1.2,<7.0.0" \
    "langchain-community>=0.4,<0.5" \
    "langchain-google-genai>=3.0.0,<4.0.0" \
    "faiss-cpu>=1.12.0,<2.0.0" \
    "uvicorn>=0.38.0,<0.39.0" \
    "python-multipart>=0.0.20,<0.0.21" 

# Python 캐시 파일 정리
RUN find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ---- 최종 이미지 ----
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 런타임 의존성 설치
RUN apt-get update && apt-get install -y \
    libffi8 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 빌드된 패키지 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 코드 복사
COPY src/ ./src/

# 사용자 설정
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "src/main.py"]