default: | help

POETRY_RUN := poetry run
PYTHON := $(POETRY_RUN) python

# -------------------------
# install
# -------------------------
.PHONY: install
install: ## install this project
	pip install poetry
	poetry install --no-dev

.PHONY: develop
develop: ## setup project for development
	pip install poetry
	poetry install

# -------------------------
# test
# -------------------------
.PHONY: unittest
unittest: ## run unit test for api with coverage
	$(PYTHON) -m pytest -v --durations=0 --cov-report=term-missing --cov=giwerm tests

.PHONY: test
test:  ## run all test
	make unittest

# -------------------------
# coding style
# -------------------------
.PHONY: lint
lint: ## type check
	$(PYTHON) -m flake8 giwerm

#typecheck: ## typing check
#	$(PYTHON) -m mypy \
#		--allow-redefinition \
#		--ignore-missing-imports \
#		--disallow-untyped-defs \
#		--warn-redundant-casts \
#		--no-implicit-optional \
#		--html-report ./mypyreport \
#		giwerm

.PHONY: format
format: ## auto format
	$(PYTHON) -m autoflake \
		--in-place \
		--remove-all-unused-imports \
		--remove-unused-variables \
		--recursive giwerm
	$(PYTHON) -m isort giwerm
	$(PYTHON) -m black \
		--line-length=119 \
		giwerm

.PHONY: pre-push
pre-push:  ## run before `git push`
	make test
	make format
	make lint


help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
