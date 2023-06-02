SRC = $(wildcard notebooks_build/*.ipynb)

all: SmilesPE docs

SmilesPE: $(SRC)
	nbdev_build_lib
	touch SmilesPE

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist