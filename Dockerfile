FROM pergrin/fictional-happiness:latest

WORKDIR /app

RUN pip install flask transformers torch tqdm

COPY . .

CMD ["python", "app.py"]
