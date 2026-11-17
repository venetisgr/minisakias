# Μηνυσάκιας: the site and its pipeline in one image. Configuration comes from the environment (see .env.example):
# TENSORX_API_KEY is the one thing it needs. The law corpus (data/laws) is baked in.
#
#   docker build -t minisakias .
#   docker run --rm -p 8000:8000 --env-file .env minisakias
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# editable, so the package stays under /app/src and finds data/ next to it
RUN pip install --no-cache-dir -e .
COPY data ./data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# a stop (a deploy) lets the answers in progress finish — below fly.toml's kill_timeout
CMD ["uvicorn", "minisakias.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "110"]
