FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# PyTorch só para CPU: imagem bem menor que a versão com CUDA.
# Para GPU, troque o índice por https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Não roda como root (arquivos já copiados com o dono certo, sem duplicar a camada dos modelos)
RUN useradd --create-home agroscope
COPY --chown=agroscope:agroscope . .
USER agroscope

ENV SERVER_PORT=5000
EXPOSE 5000

# Docker marca o contêiner como "unhealthy" se a API parar de responder
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"SERVER_PORT\"]}/health', timeout=4)"

CMD ["python", "app.py"]
