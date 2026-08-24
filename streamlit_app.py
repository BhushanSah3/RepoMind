"""Streamlit Community Cloud entry point for RepoMind."""

# ``import app`` is cached after the first Streamlit rerun, leaving the page
# empty after any interaction. Execute the source file on every rerun instead.
from pathlib import Path
from runpy import run_path

run_path(Path(__file__).with_name("app.py"), run_name="__main__")
