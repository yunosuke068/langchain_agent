SERVICE = personal_agent
YAML = docker-compose.personal_agent.yml

build:
	docker-compose -f $(YAML) build

up:
	docker-compose -f $(YAML) up -d

bash:
	docker-compose -f $(YAML) run --rm $(SERVICE) bash

down:
	docker-compose -f $(YAML) down

logs:
	docker-compose -f $(YAML) logs -f

start:
	docker-compose -f $(YAML) start

clean:
	docker-compose -f $(YAML) down --remove-orphans

clean-all:
	docker-compose -f $(YAML) down -v --remove-orphans