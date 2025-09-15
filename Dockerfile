FROM python:3.10-slim

# Set workdir inside container
WORKDIR /app

# Copy 
COPY requirements.txt .
COPY app.py .
COPY data_ingestion.py .
COPY retrieval.py . 

RUN mkdir -p data
COPY data/japantravel_posts_with_comments.csv ./data/

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-u", "app.py","--host=0.0.0.0"]