FROM python:3.13-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m venv /install \
    && /install/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.13-slim

ENV PATH="/install/bin:$PATH"
ENV HOSTNAME="Bot"
ENV USER="self"

WORKDIR /app

COPY --from=build /install /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    neofetch \
    tzdata \
    && cp /usr/share/zoneinfo/Asia/Jakarta /etc/localtime \
    && echo "Asia/Jakarta" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m self && chown -R self:self /app

COPY --chown=self:self . .

CMD ["python", "main.py"]
