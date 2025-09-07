FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Already inside src/, so just copy everything
COPY src/ .


CMD ["python", "pipeline.py"]


