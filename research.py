#!/usr/bin/env python3
"""
Arbitron research CLI entry point.

Usage:
    python research.py "Evaluate momentum in tech stocks 2021-2024"
    python research.py --pipeline --universe TECH
    python research.py --screen --universe SP500
"""

import sys
import os

def main():
    # Ensure the package is on the path when run directly
    sys.path.insert(0, os.path.dirname(__file__))
    from arbitron.agent.loop import _cli
    _cli()

if __name__ == "__main__":
    main()
