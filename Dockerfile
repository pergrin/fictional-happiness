FROM ashutoshtarun/joint_model:1

WORKDIR /app

RUN pip install flask transformers torch nltk pandas sklearn

COPY . .

CMD ["python3", "app.py"]
