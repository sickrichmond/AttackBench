# AttackBench Documentation

This directory contains the Sphinx documentation for AttackBench.

## Building the Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install from the requirements file:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

From the `docs/` directory:

```bash
cd docs/
make html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Build PDF Documentation

```bash
cd docs/
make latexpdf
```

The PDF will be in `docs/_build/latex/`.

### Clean Build Files

```bash
cd docs/
make clean
```

### Available Make Targets

- `make html` - Build HTML documentation
- `make latexpdf` - Build PDF documentation
- `make epub` - Build EPUB documentation
- `make linkcheck` - Check all external links
- `make doctest` - Run doctests in documentation
- `make clean` - Remove all build files

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── architecture.rst     # System architecture
├── optimality.rst       # Optimality metrics
├── examples.rst         # Usage examples
├── faq.rst             # Frequently asked questions
├── contributing.rst     # Contributing guide
├── api/                # API reference
│   ├── index.rst
│   ├── attacks.rst
│   ├── datasets.rst
│   ├── models.rst
│   ├── metrics.rst
│   └── analysis.rst
├── _static/            # Static files (CSS, images, etc.)
└── _templates/         # Custom templates
```

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when changes are pushed to the repository.

**Live Documentation**: https://attackbench.readthedocs.io (or your configured URL)

### Configuration

The Read the Docs build is configured in `.readthedocs.yaml` at the repository root.

## Writing Documentation

### reStructuredText (.rst) Files

Most documentation is written in reStructuredText format. Key syntax:

```rst
Section Title
=============

Subsection
----------

**Bold text**
*Italic text*
``Code``

Code blocks:

.. code-block:: python

   import attackbench
   print("Hello")

Links:

:doc:`other_page`
:ref:`section-label`
```

### API Documentation

API documentation is auto-generated from Python docstrings using Sphinx autodoc. Use Google or NumPy style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
```

### Adding New Pages

1. Create a new `.rst` file in `docs/`
2. Add it to the `toctree` in `index.rst` or relevant parent page
3. Build and check the result

## Troubleshooting

### Import Errors

If you get import errors when building, the conf.py includes mocking for heavy dependencies. Add any missing dependencies to the `MOCK_MODULES` list in `conf.py`.

### Autodoc Warnings

If autodoc can't find modules:
- Ensure the package is installed: `pip install -e .`
- Check `sys.path` configuration in `conf.py`

### Theme Issues

The documentation uses `sphinx_rtd_theme`. If you see theme errors:

```bash
pip install sphinx-rtd-theme
```

## Contributing

When adding new features, please update the relevant documentation:

1. Add API docstrings to your code
2. Update or create relevant guide pages
3. Add examples if appropriate
4. Build and test locally before committing

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs](https://docs.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
