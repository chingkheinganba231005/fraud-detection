PY ?= python

.PHONY: install data dataset train evaluate stress readme all sample test serve

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:        ## CMS Part B and Part D by-provider files for six years and the OIG list (about 6 GB)
	$(PY) scripts/01_download.py

dataset:     ## DuckDB: load, label, ratios, peer statistics, peer z-scores
	$(PY) scripts/02_build_dataset.py

train:       ## fit every detector on 2014-2017 and choose one on 2019
	$(PY) scripts/03_train.py

evaluate:    ## score 2022 once, bootstrap intervals, figures, service reference files
	$(PY) scripts/04_evaluate.py

stress:      ## synthetic fraud schemes injected into real 2022 providers
	$(PY) scripts/05_stress_test.py

readme:
	$(PY) scripts/render_readme.py

all: data dataset train evaluate stress readme

sample:      ## cut the small real sample in tests/data from the full data
	$(PY) scripts/make_sample.py

test:
	$(PY) -m pytest -q

serve:
	uvicorn fraud.service:app --app-dir src --port 8000
