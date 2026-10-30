"""Compatibility entry point: regenerate baseline and SMOTE in one consistent run."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.run_all import main

if __name__ == '__main__':
    main()
