# Hugging Face Space (Docker SDK). Serves the API on port 7860.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    STATE_DIR=/tmp/sahayak-state \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# bake the embedding model into the image so cold starts don't download 2 GB
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', allow_patterns=['*.json','*.safetensors','*.bin','sparse_linear.pt','sentencepiece.bpe.model'])"

COPY --chown=user app ./app
COPY --chown=user data/index ./data/index
COPY --chown=user data/corpus ./data/corpus

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
