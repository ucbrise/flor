PYTHON = .venv/bin/python
PIP = .venv/bin/pip

.venv:
	@python -m venv .venv
	$(PIP) install --upgrade pip; $(PIP) install -r requirements.txt

.PHONY: test
test:
	python -m pytest

# Skips the tests that spawn real forward runs and replays (~85s of the suite).
.PHONY: test-fast
test-fast:
	python -m pytest -m "not slow"

.PHONY: wheel
wheel:
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*

.PHONY:clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info