FROM ghcr.io/astral-sh/uv:0.12.19 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src ./src
RUN uv sync --locked --no-dev
ENTRYPOINT ["uv", "run", "--locked", "--no-dev", "online-shoppers"]
CMD ["--help"]
