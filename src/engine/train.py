"""
Model Training Pipeline: Calibrated XGBoost with Group-Aware Splitting
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

class ModelTrainer:
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # XGBoost parameterized for probability estimation, not hard classification
        self.base_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        
        # Isotonic calibration corrects predicted probabilities to match empirical reality
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_estimator, 
            method="isotonic", 
            cv="prefit"
        )

    def train_and_evaluate(self, df: pd.DataFrame):
        """
        Executes group-split training, calibration, and BSS evaluation.
        Expects 'fight_id' for grouping and 'bout_winner' (0 or 1) as the target.
        """
        print("Initializing Training Pipeline...")
        
        # 1. Define Features and True Target
        X = df.drop(columns=['bout_winner'])
        y = df['bout_winner']
        groups = df['fight_id']
        
        # 2. Group-Aware Splitting (Prevents Data Leakage)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Drop fight_id from feature space prior to training
        X_train_features = X_train.drop(columns=['fight_id'], errors='ignore')
        X_test_features = X_test.drop(columns=['fight_id'], errors='ignore')
        
        # 3. Train Base Model
        print(f"Training on {len(X_train)} intervals, Testing on {len(X_test)} intervals...")
        self.base_estimator.fit(X_train_features, y_train)
        
        # 4. Fit Calibrator on a validation subset (re-splitting train)
        val_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        t_idx, v_idx = next(val_gss.split(X_train_features, y_train, groups=X_train['fight_id']))
        self.calibrated_model.fit(X_train_features.iloc[v_idx], y_train.iloc[v_idx])
        
        # 5. Evaluate Brier Skill Score
        calibrated_probs = self.calibrated_model.predict_proba(X_test_features)[:, 1]
        
        bs_model = brier_score_loss(y_test, calibrated_probs)
        base_rate = y_train.mean()
        bs_baseline = brier_score_loss(y_test, [base_rate] * len(y_test))
        bss = 1 - (bs_model / bs_baseline)
        
        print("\n--- Model Evaluation (Out of Sample) ---")
        print(f"Model Brier Score:    {bs_model:.4f}")
        print(f"Baseline Brier Score: {bs_baseline:.4f} (Base Rate: {base_rate:.1%})")
        print(f"Brier Skill Score:    {bss:.4f}")
        
        if bss > 0.0:
            print("✅ Status: Model displays valid statistical signal.")
            joblib.dump(self.calibrated_model, self.model_dir / "calibrated_xgb.pkl")
        else:
            print("❌ Status: Model underperforms naive guessing. Rejecting artifact.")

        return bss

if __name__ == "__main__":
    # Example execution hook
    # df = pd.read_csv("data/processed_telemetry.csv")
    # trainer = ModelTrainer()
    # trainer.train_and_evaluate(df)
    pass