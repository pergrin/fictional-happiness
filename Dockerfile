FROM python:3

ADD app.py /

RUN pip install flask transformers torch tqdm sklearn numpy 

CMD [ "python", "./app.py" ]
