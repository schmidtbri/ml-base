import os
import sys
import json

sys.path.insert(0, os.path.abspath('../../'))

# this code exports the input and output schema info to a json schema file
# the json schema files are then used to auto generate documentation
from ml_base import __version__

# -- Project information -----------------------------------------------------
project = 'ML Base'
copyright = '2020, Brian Schmidt'
author = 'Brian Schmidt'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx']

templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
