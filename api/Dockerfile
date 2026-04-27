FROM python:3.10-slim

# ustaw katalog roboczy
WORKDIR /app

# kopiuj requirements
COPY requirements.txt .

# instaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# kopiuj resztę kodu
COPY . .

# port FastAPI
EXPOSE 8000

# start serwera
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]