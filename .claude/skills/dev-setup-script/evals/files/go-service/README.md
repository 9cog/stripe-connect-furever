# inventory-api

Small chi service that fronts the warehouse inventory feed.

## Running locally

Requires Go 1.22+ and `golangci-lint` for `make lint`.

Environment variables (all optional):

- `PORT` — listen port, default `8080`
- `FEED_URL` — upstream inventory feed, default `https://feed.northwind.internal/v2`
- `LOG_LEVEL` — `debug` | `info`, default `info`

```
make build
make run
curl localhost:8080/healthz
```
