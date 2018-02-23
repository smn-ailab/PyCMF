PYTHON ?= python

all: clean inplace

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf pycmf.egg-info
	rm -rf build

inplace:
	$(PYTHON) setup.py build_ext -i

install: clean
	$(PYTHON) setup.py install
	pytest
