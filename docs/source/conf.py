import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'skills-to-symbols'
copyright = '2024, Alper Ahmetoglu, Steve James, George Konidaris'
author = 'Alper Ahmetoglu, Steve James, George Konidaris'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']

# -- Options for autodoc extension -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': True,
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
