"""Compatibility entry point: regenerate baseline and SMOTE in one coherent run.

Training them together prevents comparisons between stale baseline and new SMOTE results.
"""
from run_pipeline import run_complete_experiments, threadpool_limits

if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run_complete_experiments()
