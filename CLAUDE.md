# Project notes for Claude

## Setup and bootstrap scripts

For any work on a setup, bootstrap or onboarding script — writing a new one, fixing or extending `setup.sh`, `bin/setup`, `scripts/*setup*` or a `make setup` target, or turning README "Getting started" steps into a script — use the `dev-setup-script` skill (`.claude/skills/dev-setup-script/`) before writing the file. It holds checks that are easy to skip: safe re-runs, a stopped Docker daemon, non-interactive migrations, version-pin conflicts, and two real test runs.

This applies to the evaluation fixtures under `.claude/skills/dev-setup-script/evals/files/` only when you're running the skill's evals; leave those repos untouched otherwise, since they're deliberately broken test inputs.
