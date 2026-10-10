FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gex_engine.py ta_engine.py signal_engine.py event_calendar.py live_market.py live_signals.py live_view.py live_dashboard.py live_forward.py .
CMD ["python", "-u", "live_forward.py"]
