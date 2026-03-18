FROM python:3.10-slim

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python seed_db.py

EXPOSE 7860

CMD ["python", "oracle.py"]