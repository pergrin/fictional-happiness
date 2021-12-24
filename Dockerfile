FROM python:3

ADD app.py /

RUN pip install flask transformers torch tqdm sklearn numpy nltk

CMD [ "python", "./app.py" ]
