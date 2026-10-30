"""Prediction exports and provenance shared by every evaluation script."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd

RESULTS = Path('results')
PROB_COLUMNS = ['P_Home', 'P_Draw', 'P_Away']
KEYS = ['MatchDate', 'Division', 'HomeTeam', 'AwayTeam']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_summary(frame):
    return {'matches': len(frame), 'start': str(frame.MatchDate.min().date()),
            'end': str(frame.MatchDate.max().date())} if len(frame) else {'matches': 0, 'start': None, 'end': None}


def holdout_label(metadata):
    part = metadata['partitions']['holdout']
    return f"Holdout ({part['start']} to {part['end']}, n={part['matches']:,})" if part['matches'] else 'Holdout (no fixtures)'


def save_predictions(name, frame, probabilities, predictions=None):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(frame), 3):
        raise ValueError('Expected one H/D/A probability vector per fixture')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('Invalid probability vectors')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate fixture identifiers')
    out = frame[KEYS + ['FTResult', 'target', 'EloDiff']].copy()
    for i, column in enumerate(PROB_COLUMNS):
        out[column] = probabilities[:, i]
    out['pred_target'] = probabilities.argmax(axis=1) if predictions is None else predictions
    out['MaxProb'] = probabilities.max(axis=1)
    out['is_correct'] = (out.target == out.pred_target).astype(int)
    path = RESULTS / 'predictions' / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format='%.17g')
    return out


def load_predictions(name):
    frame = pd.read_csv(RESULTS / 'predictions' / f'{name}.csv', parse_dates=['MatchDate'])
    p = frame[PROB_COLUMNS].to_numpy()
    if not np.allclose(p.sum(axis=1), 1) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError(f'Invalid saved probabilities: {name}')
    if frame.duplicated(KEYS).any() or not (frame.FTResult.map({'H': 0, 'D': 1, 'A': 2}) == frame.target).all():
        raise ValueError(f'Invalid fixture labels: {name}')
    if not frame.pred_target.isin([0, 1, 2]).all():
        raise ValueError(f'Invalid predicted labels: {name}')
    if not (frame.is_correct == (frame.target == frame.pred_target).astype(int)).all():
        raise ValueError(f'Incorrect saved correctness flags: {name}')
    return frame


def load_metadata():
    return json.loads((RESULTS / 'run_metadata.json').read_text())


def write_metadata(raw_path, raw_count, clean, train, test, holdout, ensemble, features):
    metadata = {
        'schema_version': 1,
        'dataset': {'file': 'data/raw/Matches.csv', 'sha256': sha256(raw_path),
                    'raw_matches': raw_count, **dataset_summary(clean)},
        'partitions': {k: dataset_summary(v) for k, v in [('train', train), ('test', test), ('holdout', holdout)]},
        'split': {'train_end_year': 2022, 'test_year': 2023, 'holdout_after_test': True},
        'ensemble': ensemble,
        'feature_columns': features,
        'environment': {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in
                        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'imbalanced-learn', 'matplotlib', 'seaborn']}},
        'source_sha256': {str(p): sha256(p) for p in sorted(Path('src').glob('*.py'))},
        'evaluation': {'threshold_selection': 'Descriptive test/holdout sweeps; no independent validation tuning claimed.',
                       'historical_holdout_2026_verified': False}
    }
    (RESULTS / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def write_manifest():
    paths = sorted(p for p in RESULTS.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json')
    manifest = {'schema_version': 1, 'sha256': {str(p): sha256(p) for p in paths},
                'readme_sha256': sha256('README.md')}
    (RESULTS / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
