# Imagem mínima: python slim (~150 MB) é suficiente, sem GPU nem libs de sistema extras.
FROM python:3.10-slim

WORKDIR /app

# Copiamos e instalamos requirements ANTES do código para aproveitar o cache de camadas
# do Docker: mudar código não reinstala as dependências.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Treina os modelos e gera relatórios durante o build: a imagem já sai pronta para
# demonstrar (o dataset vem embutido no scikit-learn, então não há download).
RUN python -m src.run_pipeline

EXPOSE 8501
# Sobe a interface. --server.address 0.0.0.0 é necessário para acessar de fora do contêiner.
CMD ["streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
