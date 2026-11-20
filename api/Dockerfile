FROM python:3.11-slim

# XGBoost necesita esta librería del sistema
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# main.py y modelo/modelo_xgboost.pkl
COPY . .

ENV PORT=8001
EXPOSE 8001
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
