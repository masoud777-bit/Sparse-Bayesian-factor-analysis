"""Make ``import bfa`` work when running pytest from the repository root."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
