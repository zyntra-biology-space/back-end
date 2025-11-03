# استخدم صورة رسمية لبايثون 3.11 (أو 3.10)
FROM python:3.11-slim

# اضبط مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات متطلبات البايثون
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ ملف server.py فقط
COPY server.py .
# COPY .env .

# اعرض البورت الذي سيعمل عليه التطبيق
EXPOSE 8000

# الأمر الافتراضي لتشغيل التطبيق باستخدام uvicorn
CMD ["python", "-m", "uvicorn", "server:app"]
