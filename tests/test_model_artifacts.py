import os
import pickle
import numpy as np


def test_model_files_and_scaler_match():
    """Ensure model artifacts exist and the saved scaler matches the scaler inside model.pkl."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = {
        'primary': os.path.join(root, 'models', 'model.pkl'),
        'scaler': os.path.join(root, 'models', 'scaler.pkl'),
        'rf': os.path.join(root, 'models', 'model_rf.pkl'),
        'xgb': os.path.join(root, 'models', 'model_xgb.pkl'),
    }

    # All files must exist
    for name, p in paths.items():
        assert os.path.exists(p), f"Missing expected artifact: {name} at {p}"

    # Load primary and scaler
    with open(paths['primary'], 'rb') as f:
        primary = pickle.load(f)
    with open(paths['scaler'], 'rb') as f:
        scaler_file = pickle.load(f)

    assert isinstance(primary, dict), "models/model.pkl should be a dict containing 'scaler'"
    primary_scaler = primary.get('scaler')
    assert primary_scaler is not None, "Primary model dict does not contain a 'scaler'"

    # Compare key scaler attributes if present
    attrs = ['mean_', 'scale_', 'var_']
    for a in attrs:
        assert hasattr(primary_scaler, a) and hasattr(scaler_file, a), f"Scaler missing attribute '{a}'"
        v1 = getattr(primary_scaler, a)
        v2 = getattr(scaler_file, a)
        assert np.allclose(v1, v2, equal_nan=True), f"Scaler attribute '{a}' differs between model.pkl and scaler.pkl"
