export IMAGE_REGISTRY ?= quay.io/sustainable_computing_io
IMAGE_NAME := kepler_model_server
IMAGE_VERSION := 0.7

IMAGE ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME):v$(IMAGE_VERSION)
LATEST_TAG_IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_NAME):latest
TEST_IMAGE := $(IMAGE)-test
MODEL_SERVER_TEST_IMAGE := $(TEST_IMAGE)-model-server
OFFLINE_TRAINER_TEST_IMAGE := $(TEST_IMAGE)-offline-trainer

CTR_CMD = docker

DOCKERFILES_PATH := ./dockerfiles
MODEL_PATH := ${PWD}/tests/models

build:
	$(CTR_CMD) build -t $(IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile .

build-test-nobase:
	$(CTR_CMD) build -t $(TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test-nobase .

build-test:
	$(CTR_CMD) build -t $(TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test .

build-test-model-server:
	$(CTR_CMD) build -t $(MODEL_SERVER_TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test-model-server .

build-test-offline-trainer:
	$(CTR_CMD) build -t $(OFFLINE_TRAINER_TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test-offline-trainer .

push:
	$(CTR_CMD) push $(IMAGE)

push-test:
	$(CTR_CMD) push $(TEST_IMAGE)

exec-test:
	$(CTR_CMD) run --platform linux/amd64 -it $(TEST_IMAGE) /bin/bash

test-pipeline:
	mkdir -p ${MODEL_PATH}
	$(CTR_CMD) run --platform linux/amd64 -v ${MODEL_PATH}:/mnt/models -i $(TEST_IMAGE) /bin/bash -c "python3.8 -u ./tests/pipeline_test.py"

# test collector --> estimator
run-estimator:
	$(CTR_CMD) run -d --platform linux/amd64 -e "MODEL_TOPURL=http://localhost:8110" -v ${MODEL_PATH}:/mnt/models -p 8100:8100 --name estimator $(TEST_IMAGE) /bin/bash -c "python3.8 tests/http_server.py & sleep 5 && python3.8 src/estimate/estimator.py"

run-collector-client:
	$(CTR_CMD) exec estimator /bin/bash -c "while [ ! -S "/tmp/estimator.sock" ]; do sleep 1; done; python3.8 -u ./tests/estimator_power_request_test.py"

clean-estimator:
	$(CTR_CMD) stop estimator
	$(CTR_CMD) rm estimator

test-estimator: run-estimator run-collector-client clean-estimator

# test estimator --> model-server
run-model-server:
	$(CTR_CMD) run -d --platform linux/amd64 -e "MODEL_TOPURL=http://localhost:8110" -v ${MODEL_PATH}:/mnt/models -p 8100:8100 --name model-server $(TEST_IMAGE) /bin/bash -c "python3.8 tests/http_server.py & sleep 10 &&  python3.8 src/server/model_server.py"
	while ! docker logs model-server | grep -q Serving; do   echo "waiting for model-server to serve";  sleep 5; done

run-model-server-prod:
	$(CTR_CMD) run -d \
	--platform linux/amd64 \
	-e "MODEL_TOPURL=http://localhost:8110" \
	-v ${MODEL_PATH}:/mnt/models \
	-p 8105:8105 \
	--name model-server-prod \
	$(MODEL_SERVER_TEST_IMAGE)

run-estimator-client:
	$(CTR_CMD) exec model-server /bin/bash -c "python3.8 -u ./tests/estimator_model_request_test.py"

clean-model-server:
	@$(CTR_CMD) stop model-server
	@$(CTR_CMD) rm model-server

clean-model-server-prod:
	@$(CTR_CMD) stop model-server-prod
	@$(CTR_CMD) rm model-server-prod

test-model-server: 
	@if [ "$(ENV)" = "prod" ]; then \
		run-model-server-prod run-estimator-client clean-model-server-prod
	else \
		run-model-server run-estimator-client clean-model-server
	fi

# test offline trainer
run-offline-trainer:
	$(CTR_CMD) run -d --platform linux/amd64 \
	-p 8102:8102 \ 
	--name offline-trainer \ 
	$(TEST_IMAGE) \ 
	python3.8 src/train/offline_trainer.py
	sleep 5

run-offline-trainer-client:
	$(CTR_CMD) exec offline-trainer \
	/bin/bash -c \
	"python3.8 -u ./tests/offline_trainer_test.py"


run-offline-trainer-client-prod:
	$(CTR_CMD) exec offline-trainer-prod \
	/bin/bash -c \
	"python3.8 -u ./tests/offline_trainer_test.py"

run-offline-trainer-prod:
	$(CTR_CMD) run -d \
	--platform linux/amd64 \
	-p 8107:8107 \
	--name offline-trainer-prod \
	$(OFFLINE_TRAINER_TEST_IMAGE)
	sleep 5

clean-offline-trainer:
	@$(CTR_CMD) stop offline-trainer
	@$(CTR_CMD) rm offline-trainer

clean-offline-trainer-prod:
	@$(CTR_CMD) stop offline-trainer-prod
	@$(CTR_CMD) rm offline-trainer-prod

test-offline-trainer: 
	@if [ "$(ENV)" = "prod" ]; then \
		run-offline-trainer-prod run-offline-trainer-client-prod clean-offline-trainer-prod
	else \
		run-offline-trainer run-offline-trainer-client clean-offline-trainer
	fi

test: build-test test-pipeline test-estimator test-model-server test-offline-trainer

# set image
set-image:
	@cd ./manifests/base && kustomize edit set image kepler_model_server=$(IMAGE)
	@cd ./manifests/server && kustomize edit set image kepler_model_server=$(IMAGE)

# deploy
_deploy:
	@$(MAKE) set-image
	@kustomize build ./manifests/base|kubectl apply -f -

# print
_print:
	@$(MAKE) set-image
	@kustomize build ./manifests/base|cat

cleanup:
	kustomize build manifests/base|kubectl delete -f -

deploy:
	@chmod +x ./manifests/set.sh
	@./manifests/set.sh "${OPTS}"
	@$(MAKE) _deploy

manifest: 
	@chmod +x ./manifests/set.sh
	@./manifests/set.sh "${OPTS}"
	@$(MAKE) _print

e2e-test:
	chmod +x ./tests/e2e_test.sh
	./tests/e2e_test.sh test "${OPTS}"

patch-power-request-client:
	kubectl patch ds kepler-exporter -n kepler --patch-file ./manifests/test/power-request-client.yaml

patch-model-request-client:
	kubectl patch ds kepler-exporter -n kepler --patch-file ./manifests/test/model-request-client.yaml