# 1. استفاده از یک ایمیج پایه پایتونی
FROM python:3.9-slim

# 2. تنظیم مسیر کاری داخل کانتینر
WORKDIR /app

# 3. کپی کردن فایل‌های مورد نیاز داخل کانتینر
COPY requirements.txt .
COPY app/ app/

# 4. نصب وابستگی‌ها
RUN pip install --no-cache-dir -r requirements.txt

# 5. اجرای برنامه (مثلاً با Streamlit)
CMD ["streamlit", "run", "app/main.py"]
