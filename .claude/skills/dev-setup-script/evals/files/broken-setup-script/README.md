# notes-app

Small Node service for team notes, backed by MongoDB.

## Setup

Run `bin/setup`, then `yarn dev` and open http://localhost:3300.

Mongo is exposed on **27018** (not the default 27017) so it doesn't clash with other projects'
databases. `docker-compose.yml` is the source of truth for the Mongo version.
