# syntax=docker/dockerfile:1

FROM python:3.12.3-slim-bullseye

WORKDIR /app

# Copy necessary files
COPY . .

RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

RUN python -m nltk.downloader stopwords

RUN python -m nltk.downloader wordnet

RUN python -m nltk.downloader punkt_tab

EXPOSE 5000

CMD [ "python", "app.py" ]