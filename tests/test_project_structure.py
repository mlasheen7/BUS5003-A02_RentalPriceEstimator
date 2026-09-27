"""
Test that the repository layout matches what the guides document.

These catch the class of problem where a fresh clone is missing a directory the
pipeline or the app expects, which is otherwise only discovered at runtime.
"""
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(REPO_ROOT, *parts)


@pytest.mark.parametrize(
    "directory",
    [
        "data/raw",
        "data/processed",
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "app/pages",
        "app/utils",
        "models",
        "notebooks",
        "reports",
        "docs",
        "tests",
    ],
)
def test_directory_exists(directory):
    """Every directory the guides reference survives a fresh clone."""
    assert os.path.isdir(_path(directory)), f"{directory}/ missing from the repository"


@pytest.mark.parametrize(
    "package",
    ["src", "src/data", "src/features", "src/models", "src/utils", "app", "app/pages", "app/utils"],
)
def test_is_importable_package(package):
    """Source directories are real Python packages, so imports work."""
    assert os.path.isfile(_path(package, "__init__.py")), f"{package}/__init__.py missing"


def test_pipeline_is_at_documented_path():
    """The pipeline lives where its own docstring and every guide say it does."""
    assert os.path.isfile(_path("src", "data", "pipeline.py"))


def test_env_template_is_copyable():
    """`cp .env.example .env` in the setup guide has something to copy."""
    assert os.path.isfile(_path(".env.example"))


def test_env_file_is_not_committed():
    """A real .env must never be tracked, even if one exists locally."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tracked == "", ".env is tracked by git - remove it from the index"


def test_processed_data_is_present():
    """data/processed/ is committed; the model work depends on it."""
    for name in [
        "merged_rental_data.csv",
        "X_train.pkl",
        "y_train.pkl",
        "feature_names.txt",
    ]:
        assert os.path.isfile(_path("data", "processed", name)), f"data/processed/{name} missing"
