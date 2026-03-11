FROM python:3.12-slim

LABEL org.opencontainers.image.title="MT5 MCP Server"
LABEL org.opencontainers.image.description="Portable MCP server for MetaTrader 5 trading via com.lattice24/mt5-trading"
LABEL org.opencontainers.image.vendor="lattice24"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server.py .

ENV MT5_BRIDGE_URL=http://host.docker.internal:8787
ENV MT5_BRIDGE_API_KEY=

ENTRYPOINT ["python", "mcp_server.py"]
