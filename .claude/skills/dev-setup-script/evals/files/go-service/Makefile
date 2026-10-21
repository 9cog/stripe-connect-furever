BIN := bin/inventory-api

.PHONY: build run test lint

build:
	go build -o $(BIN) ./cmd/api

run: build
	./$(BIN)

test:
	go test ./...

lint:
	golangci-lint run
