# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('..'))

# -- Mock heavy dependencies for building docs -------------------------------
# This prevents ImportErrors when building docs without installing all dependencies
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = [
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils',
    'torch.utils.data', 'torchvision', 'torchvision.transforms', 'torchvision.datasets',
    'torchvision.models', 'foolbox', 'robustbench', 'cleverhans', 'deeprobust',
    'art', 'art.attacks', 'art.estimators', 'torchattacks', 'adv_lib',
    'timm', 'transformers', 'pretrainedmodels', 'sacred', 'wandb',
    'sklearn', 'seaborn', 'plotly', 'scipy', 'scipy.spatial'
]

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AttackBench'
copyright = '2026, Antonio Emanuele Cinà, Riccardo Trebiani'
author = 'Antonio Emanuele Cinà, Riccardo Trebiani'
version = '1.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    # 'sphinx_autodoc_typehints',  # Temporarily disabled due to compatibility issues with mocked modules
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master toctree document
master_doc = 'index'

# Add any paths that contain custom static files (such as style sheets)
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Language
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# GitHub integration
html_context = {
    'display_github': True,
    'github_user': 'attackbench',
    'github_repo': 'AttackBench',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

html_title = f"{project} {version} documentation"
html_short_title = project
html_logo = None
html_favicon = None

# Custom sidebar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_mock_imports = MOCK_MODULES

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Napoleon settings (for Google and NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Todo extension
todo_include_todos = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
