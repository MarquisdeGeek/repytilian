import subprocess
import unittest

class TestMypy(unittest.TestCase):
    def test_training(self):
        """Run mypy on the source code and fail if there are type errors."""
        result = subprocess.run(
            ["mypy", "training.py"], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Mypy found type errors:\n{result.stdout}"

    def test_generate(self):
        """Run mypy on the source code and fail if there are type errors."""
        result = subprocess.run(
            ["mypy", "generate.py"], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Mypy found type errors:\n{result.stdout}"
