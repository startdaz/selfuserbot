FROM python:3.11-alpine AS build

RUN apk add --no-cache python3-dev gcc musl-dev libffi-dev openssl-dev

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /install \
    && /install/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-alpine

ENV PATH="/install/bin:$PATH"
ENV HOSTNAME="Bot"

WORKDIR /app

COPY --from=build /install /install

RUN apk add --no-cache tzdata \
    && cp /usr/share/zoneinfo/Asia/Jakarta /etc/localtime \
    && echo "Asia/Jakarta" > /etc/timezone \
    && adduser -D self \
    && chown -R self:self /app

COPY --chown=self:self . .

USER self

CMD ["python", "main.py"]
