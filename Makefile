
.PHONY: base-venv
base-venv:
	python3.10 -m venv .venv
	. .venv/bin/activate;
	pip install --upgrade pip
	pip install pip-tools

requirements.txt: requirements.in
	pip-compile --generate-hashes requirements.in

.PHONY: install
install: requirements.txt
	. .venv/bin/activate; pip install -r requirements.txt

.PHONY: clean-venv
clean-venv:
	rm -rf .venv


# run docker-compose from the app directory
.PHONY: streamlit-app
streamlit-app:
	docker-compose -f app/docker-compose.yaml up --build

.PHONY: streamlit-app-down
streamlit-app-down:
	docker-compose -f app/docker-compose.yaml down

# clear all docker containers and volumes
.PHONY: streamlit-app-clear
streamlit-app-clear:
	docker-compose -f app/docker-compose.yaml down --volumes