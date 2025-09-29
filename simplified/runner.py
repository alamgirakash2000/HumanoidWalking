import os
import sys

# Ensure project root on path when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from simplified.g1_benchmark_pid.runner import main

if __name__ == "__main__":
	main()
