from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import numpy as np


@dataclass
class Phase4Config:
    """پیکربندی جامع فاز ۴ - مدل‌سازی و ارزیابی"""

    # تنظیمات پایه
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_JOBS: int = -1

    # تنظیمات اعتبارسنجی
    CV_STRATEGY: str = 'stratified'
    N_SPLITS: int = 5

    # مدل‌های پایه
    BASE_MODELS: Dict[str, Any] = None

    # هایپرپارامترهای تنظیم
    HYPERPARAM_GRIDS: Dict[str, Dict[str, Any]] = None

    # معیارهای ارزیابی
    EVALUATION_METRICS: List[str] = None

    # مسیرهای ذخیره‌سازی
    MODEL_SAVE_PATHS: Dict[str, str] = None

    def __post_init__(self):
        if self.BASE_MODELS is None:
            self.BASE_MODELS = {
                'logistic_regression': {
                    'class': 'LogisticRegression',
                    'params': {
                        'random_state': self.RANDOM_STATE,
                        'max_iter': 1000,
                        'n_jobs': self.N_JOBS
                    }
                },
                'knn': {
                    'class': 'KNeighborsClassifier',
                    'params': {
                        'n_jobs': self.N_JOBS
                    }
                },
                'svm': {
                    'class': 'SVC',
                    'params': {
                        'random_state': self.RANDOM_STATE,
                        'probability': True
                    }
                },
                'random_forest': {
                    'class': 'RandomForestClassifier',
                    'params': {
                        'random_state': self.RANDOM_STATE,
                        'n_jobs': self.N_JOBS
                    }
                }
            }

        if self.HYPERPARAM_GRIDS is None:
            self.HYPERPARAM_GRIDS = {
                'svm': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced']
                },
                'random_forest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced']
                }
            }

        if self.EVALUATION_METRICS is None:
            self.EVALUATION_METRICS = [
                'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro'
            ]

        if self.MODEL_SAVE_PATHS is None:
            self.MODEL_SAVE_PATHS = {
                'base_models': 'data/models/trained_models',
                'tuned_models': 'data/models/tuned_models',
                'preprocessors': 'data/models/preprocessors',
                'evaluation': 'data/models/evaluation_results'
            }

    def get_cv_strategy(self):
        """دریافت استراتژی اعتبارسنجی متقابل"""
        if self.CV_STRATEGY == 'timeseries':
            return TimeSeriesSplit(n_splits=self.N_SPLITS)
        else:
            return StratifiedKFold(n_splits=self.N_SPLITS, shuffle=True,
                                   random_state=self.RANDOM_STATE)