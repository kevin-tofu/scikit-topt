# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import shutil
import sys
sys.path.insert(0, os.path.abspath('../scikit-topt'))


project = 'scikit-topt'
copyright = '2025, Kohei Watanabe'
author = 'Kohei Watanabe'
release = '0.2.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    "sphinx_sitemap",
    'myst_parser'
]
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = ["extra"]
html_meta = {
    "description": "scikit-topt(sktopt) is a scientific topology optimization library in Python.",
    "keywords": "topology optimization, FEM, Python, scientific computing",
    "author": "Kohei Watanabe",
    "robots": "index, follow",
}

html_baseurl = 'https://kevin-tofu.github.io/scikit-topt/'
