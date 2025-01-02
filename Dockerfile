FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt

RUN pip install flask
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_and_infer.py"]