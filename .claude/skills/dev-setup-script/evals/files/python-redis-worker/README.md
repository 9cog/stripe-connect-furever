# thumbgen

Pulls jobs off the `thumbs` Redis queue and writes resized images to `THUMB_OUTPUT_DIR`.

## Setup

You need Python 3.11 and a local Redis. On a Mac: `brew install redis && brew services start redis`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
mkdir -p out
make worker
```

Run `make test` before opening a PR.
