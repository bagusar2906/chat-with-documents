FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "agent-service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
