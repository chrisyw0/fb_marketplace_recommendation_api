FROM --platform=linux/amd64 python:3.9-slim as builder
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
build-essential gcc git

WORKDIR app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirement.txt .

RUN pip install --upgrade pip
RUN pip install -r requirement.txt

FROM --platform=linux/amd64 python:3.9-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR app
COPY --from=builder /opt/venv /opt/venv
COPY ./src .
COPY ./model ./model

ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 80

CMD ["uvicorn", "main:app"]