# Development

## Install the library locally

Use `pip install -e` to create an **editable installation**, by first cloning the repository and then:

      pip install -e <path to gradvar>
    
This creates a symbolic link to your library directory, so changes in the library are immediately reflected in any project that uses it.

## Build the library

```
pip install -r requirements-dev.txt
pip install -r requirements.txt
python -m build
```

## Upload to PyPi

```
twine upload dist/*
```

## Release WHL file to Github

```
conda install gh --channel conda-forge
gh auth login
gh release create release-2025.3.1 dist/gradvar-2025.3.1-py3-none-any.whl --generate-notes
```
