# build everything
all: test deployment

install:
	python3 -m pip install .

# run the Python test suite
test: install
	python3 -m unittest discover .

# clean the build directory
clean:
	rm -rf build/ dist/ .eggs/ *.egg-info/ || true
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# build the deployment package
deployment: clean
	python3 -m build --sdist --wheel

# ship the deployment package to PyPi
ship: test deployment
	twine upload dist/*
