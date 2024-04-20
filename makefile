docker-build:
	docker build -t pipeline -f docker/Dockerfile .

docker-run:
	docker run -it --rm -v ./:/pipeline/ --gpus all --network=host pipeline