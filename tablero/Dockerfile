FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# código, datos agregados, catálogos, métricas y estilos (el modelo vive en la API)
COPY . .

ENV PORT=8050
EXPOSE 8050
# API_URL y API_PORT se pasan al ejecutar el contenedor (-e API_URL=... -e API_PORT=8001)
CMD ["sh", "-c", "gunicorn app:server -b 0.0.0.0:${PORT} --workers 2 --timeout 120"]
