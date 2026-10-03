FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py .

COPY data/processed/analytics_product ./data/processed/analytics_product
COPY data/processed/analytics_company ./data/processed/analytics_company
COPY data/processed/analytics_state ./data/processed/analytics_state
COPY data/processed/analytics_monthly ./data/processed/analytics_monthly
COPY data/processed/analytics_response ./data/processed/analytics_response
COPY data/processed/analytics_timely ./data/processed/analytics_timely
COPY data/processed/analytics_issue ./data/processed/analytics_issue

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]