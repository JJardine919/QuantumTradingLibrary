FROM python:3.12-slim

LABEL org.opencontainers.image.title="Voodoo"
LABEL org.opencontainers.image.description="Voodoo AOI - Browser + Chat + 24D Octonion Brain"

# Playwright system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    fonts-liberation libappindicator3-1 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install chromium

COPY aoi_collapse.py .
COPY voodoo_service.py .

ENV OLLAMA_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=mistral
ENV VOODOO_PORT=8777

EXPOSE 8777

CMD ["python", "voodoo_service.py"]
