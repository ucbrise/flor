.PHONY: wheel
wheel:
	python setup.py sdist bdist_wheel
	python -m twine upload dist/*

.PHONY:clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info