# syntax=docker/dockerfile:1
# CPU image of the server. The tiny random model is the default so the image
# proves itself without any download; point --model at a Hugging Face Llama-family
# checkpoint (and mount your HF cache) for real text.
#
#   docker build -t pagedbatch .
#   docker run --rm -p 8000:8000 pagedbatch
#   docker run --rm -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
#       pagedbatch serve --model HuggingFaceTB/SmolLM2-135M-Instruct --host 0.0.0.0
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY pagedbatch/ pagedbatch/
RUN pip install --no-cache-dir ".[hub]"
EXPOSE 8000
ENTRYPOINT ["pagedbatch"]
CMD ["serve", "--model", "tiny", "--host", "0.0.0.0", "--port", "8000"]
