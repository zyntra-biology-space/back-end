# استخدم صورة رسمية لبايثون 3.11 (أو 3.10)
FROM python:3.11-slim

# اضبط مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات متطلبات البايثون
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل الملفات للمجلد العمل
COPY . .

# تعيين متغيرات البيئة المهمة (مثال فقط)
ENV MONGO_URI="mongodb+srv://infocodivera_db_user:m6Uwjdv2f53imWeJ@cluster0.ldqe96m.mongodb.net/?retryWrites=true&w=majority"
ENV PINECONE_API_KEY="pcsk_RT6wY_N5JbiUjPaTaLDaxXepgh7uPXpKj7wmiJVAjARHPc2HzDodSnPRKTRRZpCyEoKzh"
ENV GENAI_API_KEY="AIzaSyC7qMkLOLrCpaV6-XrdZWcvmOs4ugF3xFc"

# اعرض البورت الذي سيعمل عليه التطبيق
EXPOSE 8000

# الأمر الافتراضي لتشغيل التطبيق باستخدام uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
