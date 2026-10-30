"""One command for fitting, diagnostics, documentation and provenance."""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    from src.run_pipeline import run_complete_experiments
    from src.error_analysis import run_error_analysis
    from src.plot_roc_curves import generate_all_roc_curves
    from src.plot_feature_importance import plot_importances
    from src.update_readme import update_readme
    from src.artifacts import write_manifest
    run_complete_experiments(include_smote=True)
    run_error_analysis()
    generate_all_roc_curves()
    plot_importances()
    update_readme()
    write_manifest()
    print('Completed one canonical run. See results/run_metadata.json and results/artifact_manifest.json.')


if __name__ == '__main__':
    main()
