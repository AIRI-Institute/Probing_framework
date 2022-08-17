FROM python:3.8-buster
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD python -m uvicorn backend:app --host 0.0.0.0 --port 8000

