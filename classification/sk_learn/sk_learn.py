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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.config import update_config
from utils.preprocess_signals import pad_sequences, replace_nan_with_first_value

TEST_FILE_NAME = "test_only_2_5_10.json"
RANDOM_STATE = 42

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


    # --- Load TRAIN data ---
    print("Loading Train data...")
    train_path = os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.TRAIN_FILE_NAME)
    train_data, train_labels, train_skier_ids, max_length = load_dataset(
        train_path, cfg.DATA_PRESET.CHOOSEN_JOINTS)

    # Flatten for sklearn
    X_train = train_data.reshape(train_data.shape[0], -1)
    y_train = train_labels

    print(f"Training data shape: {X_train.shape}")
    
    # --- Load TEST data ---
    print("Loading Test data...")
    test_path = os.path.join(cfg.DATASET.ROOT_PATH, TEST_FILE_NAME)
    test_data, test_labels, test_skier_ids, _ = load_dataset(
        test_path, cfg.DATA_PRESET.CHOOSEN_JOINTS, max_length=max_length)
    
    X_test = test_data.reshape(test_data.shape[0], -1)
    y_test = test_labels

    print(f"Test data shape: {X_test.shape}")
    
    # Define models to compare
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=RANDOM_STATE)),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE))
    }
    
    print("\n--- Model Comparison ---")
    print(f"{'Model':25s} {'Train Acc':>12s} {'Test Acc':>12s}")
    print("-" * 50)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        print(f"{name:25s} {train_acc*100:10.2f}% {test_acc*100:10.2f}%")

    print("-" * 50)
    
    # with Crossvalidation
    # Cross-validation setup => not sure if crossval is necessary in this case
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- Cross-Validation Results (5-fold) ---")
    print(f"{'Model':25s} {'CV Mean Acc':>12s} {'CV Std':>10s} {'Test Acc':>12s}")
    print("-" * 60)

    for name, clf in models.items():
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"{name:25s} {np.mean(cv_scores)*100:10.2f}% {np.std(cv_scores)*100:8.2f}% {test_acc*100:10.2f}%")

    print("-" * 60)
        
    

    # # Train SVM
    # clf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    # clf.fit(X_train, y_train)
    # acc = clf.score(X_train, y_train)
    # print(f"Train accuracy: {acc:.3f}")

    # # Evaluate
    # acc = clf.score(X_test, y_test)
    # print(f"Test accuracy: {acc:.3f}")

if __name__ == '__main__':
    main()
