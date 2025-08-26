This docs site was made using the following:
- mkdocs: https://www.mkdocs.org/
- mkdocstrings: https://mkdocstrings.github.io/
- mkdocs material theme: https://squidfunk.github.io/mkdocs-material/
- pymdown-extensions: https://facelessuser.github.io/pymdown-extensions/

Automatic code documentations assumes google-style docstrings. 
For examples on how to format google-style docstrings, see here: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

### To install
Run
```python
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

### To run
From docs folder run:
```python
. .venv/bin/activate
mkdocs serve
```