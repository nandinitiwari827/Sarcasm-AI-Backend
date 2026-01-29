FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 🔥 UNZIP SRC
RUN unzip src.zip && rm src.zip

EXPOSE 7860

CMD ["python", "app.py"]