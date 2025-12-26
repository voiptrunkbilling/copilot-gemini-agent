"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file."""
    def _create(name: str, content: str = "") -> Path:
        path = temp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    return _create
