init:
	python3 -m ensurepip
	python -m venv ./.venv
	.venv/Scripts/activate
    pip install -r requirements.txt

test:
    py.test tests

run:
	python3 ./app/run.py

.PHONY: init test run