FROM python:3.9.16-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libgomp1

COPY requirements-reporting.txt .

RUN pip install --no-cache-dir prefect==2.10.9
RUN pip install --no-cache-dir prefect-gcp==0.4.2

RUN pip install -r requirements-reporting.txt

RUN prefect config set PREFECT_UI_API_URL="http://prefect_server:4200/api"
RUN prefect config set PREFECT_API_URL="http://prefect_server:4200/api"

RUN prefect config view

WORKDIR /app

COPY ["./generate_evidently_report.py", "./"]

# Start the agent and then run the main script to add work to the queue
ENTRYPOINT sleep 10 && prefect agent start --pool default-agent-pool --work-queue evidently_report_queue