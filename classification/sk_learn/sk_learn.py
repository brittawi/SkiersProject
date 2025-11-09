# Ensure project root is in sys.path
# Need to go two paths up here
import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.signal import resample

from utils.config import update_config
from utils.preprocess_signals import pad_sequences, replace_nan_with_first_value

TEST_FILE_NAME = "test_full.json"

def load_dataset(json_path, chosen_joints, pad_value=float("nan"), max_length=None):
    with open(json_path, "r") as f:
        data_json = json.load(f)

    data, labels, skier_ids = [], [], []
    for cycle in data_json.values():
        cycle_data = [np.array(cycle[joint], dtype=np.float32)
                      for joint in chosen_joints]
        cycle_tensor = np.stack(cycle_data)  # (num_joints, time_steps)
        data.append(cycle_tensor)
        labels.append(cycle["Label"])
        skier_ids.append(cycle["Skier_id"])

    # Determine pad length
    if max_length is None:
        max_length = max(seq.shape[1] for seq in data)

    data = pad_sequences(data, max_length=max_length, pad_value=pad_value)
    data = replace_nan_with_first_value(data)

    return np.array(data), np.array(labels), np.array(skier_ids), max_length


def main():
    print("Loading config...")
    cfg = update_config("./classification/training/config.yaml")


    # --- TRAIN ---
    print("Loading Train data...")
    train_path = os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.TRAIN_FILE_NAME)
    train_data, train_labels, train_skier_ids, max_length = load_dataset(
        train_path, cfg.DATA_PRESET.CHOOSEN_JOINTS)

    # Flatten for sklearn
    X_train = train_data.reshape(train_data.shape[0], -1)
    y_train = train_labels

    print(f"Training data shape: {X_train.shape}")

    # Train SVM
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    clf.fit(X_train, y_train)

    # --- TEST ---
    print("Loading Test data...")
    test_path = os.path.join(cfg.DATASET.ROOT_PATH, TEST_FILE_NAME)
    test_data, test_labels, test_skier_ids, _ = load_dataset(
        test_path, cfg.DATA_PRESET.CHOOSEN_JOINTS, max_length=max_length)
    
    X_test = test_data.reshape(test_data.shape[0], -1)
    y_test = test_labels

    print(f"Test data shape: {X_test.shape}")

    # Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")

if __name__ == '__main__':
    main()
