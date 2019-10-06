APP_NAME=ngxbac/pytorch_cv:kaggle_cell
CONTAINER_NAME=kaggle_cell
DATA_DIR=/raid/data/kaggle/recursion-cell
OUT_DIR=/raid/bac/kaggle/logs/recursion-cell

run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v $(DATA_DIR):/data \
		-v $(OUT_DIR):/logs \
		-v $(shell pwd):/kaggle-cell $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}