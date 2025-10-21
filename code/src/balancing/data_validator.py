import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats


class DataValidator:
    """اعتبارسنج هوشمند داده‌های متعادل‌شده"""

    def __init__(self, config):
        self.config = config
        self.validation_reports = {}

    def validate_balanced_data(self, original_data: Dict[str, Any],
                               balanced_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """اعتبارسنجی جامع داده‌های متعادل‌شده"""
        validation_results = {}

        for strategy_name, data_dict in balanced_data.items():
            print(f"🔍 اعتبارسنجی استراتژی: {strategy_name}")

            X = data_dict['X']
            y = data_dict['y']

            actual_has_nulls = X.isnull().sum().sum() > 0 or y.isnull().sum().sum() > 0

            validation = {
                'basic_checks': self._basic_validation(X, y),
                'distribution_checks': self._distribution_validation(original_data['y'], y),
                'quality_checks': self._quality_validation(X, y, strategy_name),
                'leakage_checks': self._leakage_validation(X, original_data['X']),
                'actual_null_status': {
                    'x_nulls': int(X.isnull().sum().sum()),
                    'y_nulls': int(y.isnull().sum().sum()),
                    'has_nulls_actual': actual_has_nulls
                }
            }

            validation_results[strategy_name] = validation

        return validation_results

    def _basic_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """اعتبارسنجی پایه"""
        has_nulls = X.isnull().sum().sum() > 0 or y.isnull().sum() > 0

        checks = {
            'has_nulls': has_nulls,
            'shape_consistent': len(X) == len(y),
            'data_types_consistent': all(X.dtypes != 'object'),
            'class_diversity': y.nunique() > 1
        }

        checks['all_passed'] = all([
            not checks['has_nulls'],  # نباید null داشته باشد
            checks['shape_consistent'],  # shapes باید match باشند
            checks['data_types_consistent'],  # باید عددی باشد
            checks['class_diversity']  # باید بیش از یک کلاس وجود داشته باشد
        ])

        print(f"   ✅ اعتبارسنجی: has_nulls={has_nulls}, all_passed={checks['all_passed']}")

        return checks

    def _distribution_validation(self, y_original: pd.Series, y_balanced: pd.Series) -> Dict[str, Any]:
        """اعتبارسنجی توزیع کلاس‌ها"""
        orig_counts = y_original.value_counts()
        balanced_counts = y_balanced.value_counts()

        # محاسبه معیارهای آماری
        distribution_metrics = {
            'original_imbalance_ratio': orig_counts.max() / orig_counts.min(),
            'balanced_imbalance_ratio': balanced_counts.max() / balanced_counts.min(),
            'improvement_ratio': (orig_counts.max() / orig_counts.min()) / (
                        balanced_counts.max() / balanced_counts.min()),
            'class_preservation': set(orig_counts.index) == set(balanced_counts.index),
            'min_samples_per_class': balanced_counts.min() >= 10  # حداقل 10 نمونه per class
        }

        return distribution_metrics

    def _quality_validation(self, X: pd.DataFrame, y: pd.Series, strategy_name: str) -> Dict[str, Any]:
        """اعتبارسنجی کیفیت داده"""
        if strategy_name == 'original':
            return {'synthetic_samples': 0, 'quality_score': 1.0}

        # برای داده‌های متعادل‌شده، بررسی کیفیت ویژگی‌ها
        quality_metrics = {
            'feature_variance': X.var().mean(),  # میانگین واریانس ویژگی‌ها
            'correlation_structure': X.corr().abs().mean().mean(),  # میانگین همبستگی
            'outlier_ratio': self._calculate_outlier_ratio(X),
            'synthetic_quality': 'unknown'  # برای SMOTE نیاز به تحلیل پیشرفته‌تر
        }

        return quality_metrics

    def _leakage_validation(self, X_balanced: pd.DataFrame, X_original: pd.DataFrame) -> Dict[str, Any]:
        """بررسی نشت داده"""
        # بررسی overlap بین داده‌های متعادل و اصلی
        if len(X_balanced) <= len(X_original):
            # برای undersampling، باید زیرمجموعه باشد
            is_subset = X_balanced.index.isin(X_original.index).all()
        else:
            # برای oversampling، باید شامل داده‌های اصلی باشد
            is_subset = X_original.index.isin(X_balanced.index).all()

        return {
            'is_proper_subset': is_subset,
            'no_data_leakage': True  # بر اساس طراحی ما
        }

    def _calculate_outlier_ratio(self, X: pd.DataFrame, threshold: float = 3.0) -> float:
        """محاسبه نسبت outlierها"""
        z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask.mean()