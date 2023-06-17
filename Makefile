LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# Run:
#  'make help' to see commands

# Requires:
#   make
#   docker 
#   docker-compose

help:
	@echo "\nCommands:\n"
	@cat Makefile | egrep -e '^#run:.*'| sed -e 's~#~~g'
# @make -qpRr | egrep -e '^[a-z].*:' | sed -e 's~:~~g' | sort
	@echo ""

#run: make model-train-flow   to apply the automated model training DAG
model-train-flow:
	docker exec prefect_agent1 python3 /app/main.py

#run: make evidently-report-flow   to apply the automated model training DAG 
evidently-report-flow:
	docker exec reporting_agent python3 /app/generate_evidently_report.py

