# syntax=docker/dockerfile:1

FROM python:3.9.16-slim

WORKDIR /app

COPY requirements.txt .
COPY app_UI_predictions.py .

RUN pip3 install -r requirements.txt
RUN pip3 install evidently==0.1.51.dev0

RUN apt update
RUN apt install -y libgomp1

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app_UI_predictions:server"]