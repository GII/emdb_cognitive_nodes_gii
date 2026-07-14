# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath("../../cognitive_nodes"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "e-MDB Cognitive Nodes implemented by the GII"
copyright = "2025, GII"
author = "GII"
release = "Apache-2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "rclpy",
    "tensorflow",
    "tf",
    "std_msgs",
    "core",
    "core_interfaces",
    "cognitive_processes",
    "cognitive_processes_interfaces",
    "cognitive_node_interfaces",
    "simulators",
    "simulators_interfaces",
    "sklearn",       
    "numpy",
    "pandas",           
    "scipy",
    "torch",           
    "geometry_msgs",  
    "sensor_msgs",    
    "nav_msgs",      
    "builtin_interfaces"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
