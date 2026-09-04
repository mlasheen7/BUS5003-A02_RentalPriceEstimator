"""
Test that the development environment is properly set up.
"""
import sys
import importlib


def test_pandas_installed():
    """Verify pandas is installed."""
    assert importlib.util.find_spec("pandas") is not None


def test_xgboost_installed():
    """Verify xgboost is installed."""
    assert importlib.util.find_spec("xgboost") is not None


def test_anthropic_installed():
    """Verify anthropic (Claude API) is installed."""
    assert importlib.util.find_spec("anthropic") is not None


def test_data_directory_exists(tmp_path):
    """Verify data directory structure exists."""
    import os
    assert os.path.exists("data/raw"), "data/raw directory missing"
    assert os.path.exists("data/processed"), "data/processed directory missing"


if __name__ == "__main__":
    test_pandas_installed()
    test_xgboost_installed()
    test_anthropic_installed()
    test_data_directory_exists(None)
    print("✅ All setup tests passed!")