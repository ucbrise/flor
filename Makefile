.PHONY: wheel
wheel:
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*
	python -m pip install -e .

.PHONY:clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info