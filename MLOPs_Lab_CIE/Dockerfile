FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY data/ data/
COPY models/ models/
COPY src/predict_cli.py src/predict_cli.py
COPY results/ results/

ENTRYPOINT ["python", "src/predict_cli.py"]