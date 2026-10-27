FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only: imagem bem menor que a versão com CUDA. Para GPU, troque o índice por https://download.pytorch.org/whl/cu130
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
