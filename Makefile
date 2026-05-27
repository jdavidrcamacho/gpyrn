.DEFAULT_GOAL := help

.PHONY: help
help:
		@echo "Please use 'make <target>' where <target> is one of:"
		@echo ""
		@echo " format-check            runs the formatting tools, only checking for errors"
		@echo " format-fix              runs the formatting tools, fixing errors where possible"
		@echo " lint-check              runs the linting tools, only checking for errors"
		@echo " type-check              runs the static type checker, only checking for errors"
		@echo " unit-test               runs the unit tests"
		@echo " docstring-check         runs the docstring checker"
		@echo ""

.PHONY: format-check
format-check:
	black --check src/gpyrn/ examples/ tests/
	isort --check-only src/gpyrn/ examples/ tests/

.PHONY: format-fix
format-fix:
	black src/gpyrn/ examples/ tests/
	isort src/gpyrn/ examples/ tests/

.PHONY: lint-check
lint-check:
	flake8 --statistics src/gpyrn/ examples/ tests/

.PHONY: type-check
type-check:
	mypy src/gpyrn/ tests/

.PHONY: unit-test
unit-test:
	pytest --cov-report term-missing --cov=gpyrn -vv

.PHONY: docstring-check
docstring-check:
	pydocstyle src/gpyrn/
