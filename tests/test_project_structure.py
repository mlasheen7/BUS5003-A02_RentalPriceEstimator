"""
Guard the repository layout and the conventions in CONTRIBUTING.md.

These run in CI on every pull request, so a broken clone is caught before merge
rather than on a teammate's machine.
"""
import os
import importlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def test_source_packages_are_importable():
    """src/ and its subpackages must be importable for --cov=src to work."""
    for module in ("src", "src.data", "src.features", "src.models", "src.utils"):
        assert importlib.util.find_spec(module) is not None, f"{module} not importable"


def test_data_directories_survive_a_fresh_clone():
    """data/raw/ is gitignored, so it needs a tracked .gitkeep to exist on clone."""
    assert os.path.isdir(_path("data", "raw")), "data/raw missing"
    assert os.path.isfile(_path("data", "raw", ".gitkeep")), (
        "data/raw/.gitkeep missing - data/raw will not exist in a fresh clone"
    )
    assert os.path.isdir(_path("data", "processed")), "data/processed missing"


def test_env_template_is_present_and_named_correctly():
    """SETUP_GUIDE tells people to run `cp .env.example .env`."""
    assert os.path.isfile(_path(".env.example")), ".env.example missing"


def test_real_env_file_is_not_committed():
    """A committed .env would leak API keys."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", ".env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tracked == "", ".env is tracked by Git - remove it and rotate the keys"


def test_pipeline_lives_where_the_guides_say_it_does():
    assert os.path.isfile(_path("src", "data", "pipeline.py")), (
        "src/data/pipeline.py missing - every guide references this path"
    )
