FROM python:3.13-alpine

ENV TZ=Asia/Jakarta

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
