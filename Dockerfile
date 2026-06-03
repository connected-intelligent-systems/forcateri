FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /forcateri

COPY . .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt
RUN python3.12 -m pip install .
