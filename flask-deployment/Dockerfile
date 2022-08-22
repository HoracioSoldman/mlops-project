FROM python:3.9.7-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -q -r requirements.txt

COPY [ "delay_predictor.py", "lin_reg.bin", "./" ]

EXPOSE 7200

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:7200", "delay_predictor:app" ]