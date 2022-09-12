FROM --platform=linux/amd64 python:3.9

COPY . .

RUN pip install --upgrade pip
RUN pip3 install -r requirement.txt

CMD ["uvicorn", "main:app", "--reload"]