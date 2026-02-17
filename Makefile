.PHONY: install sanity run run-web clean test

install:
	pip install -r requirements.txt

sanity:
	python chatbot.py sanity

run:
	python chatbot.py

run-web:
	python app.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf rag_index/
	rm -f artifacts/sanity_output.json
	rm -rf __pycache__ .pytest_cache
	find . -type f -name "*.pyc" -delete