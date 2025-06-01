FROM python:3.13-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock /app/
RUN uv sync --no-dev
COPY ./app.py /app/
COPY ./static /app/static/
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
