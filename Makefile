wheel:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*
	python3 -m pip install -e .

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info