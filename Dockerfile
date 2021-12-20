FROM python:3.9.5-slim-buster

ENV PYTHONUNBUFFERED 1

WORKDIR /app

ADD requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

ADD app.py ./app.py
COPY music.csv ./music.csv
COPY bm25.pkl ./bm25.pkl
COPY .streamlit /root/.streamlit

EXPOSE 80

CMD [ "streamlit", "run", "app.py" ]