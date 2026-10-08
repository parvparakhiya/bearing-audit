FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-lock.txt ./
RUN pip install -r requirements-lock.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

COPY data/manifest.csv data/PROVENANCE.md ./data/
RUN useradd --create-home --uid 1000 app && mkdir -p data/raw reports && chown -R app /app
USER app

ENTRYPOINT ["bearing-audit"]
CMD ["--help"]
