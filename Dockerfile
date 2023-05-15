# syntax=docker/dockerfile:1

FROM python:3.9

WORKDIR /code

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE $PORT

ENTRYPOINT ["gunicorn", "app:app", "--bind", ":$PORT"]

