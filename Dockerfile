# pandas 3.x exige Python 3.11 ou superior.
FROM python:3.12-slim

WORKDIR /app

# Dependências instaladas antes da cópia do código para reaproveitar a camada em cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Treina os modelos e gera os relatórios durante o build.
RUN python -m src.run_pipeline

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
