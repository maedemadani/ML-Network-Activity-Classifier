import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings('ignore')


class AdaptiveSamplingStrategies:
    """استراتژی‌های نمونه‌برداری پویا و هوشمند"""

    def __init__(self, config):
        self.config = config
        self.samplers = {}
        self.sampling_reports = {}

    def calculate_dynamic_parameters(self, y_train: pd.Series) -> Dict[str, Any]:
        """محاسبه پارامترهای نمونه‌برداری بر اساس توزیع واقعی داده"""
        value_counts = y_train.value_counts()
        class_names = value_counts.index.tolist()

        # محاسبه آمارهای مختلف
        majority_class = value_counts.idxmax()
        majority_count = value_counts.max()
        minority_class = value_counts.idxmin()
        minority_count = value_counts.min()
        median_count = int(value_counts.median())
        mean_count = int(value_counts.mean())

        # انتخاب استراتژی بر اساس نوع عدم تعادل
        imbalance_ratio = majority_count / minority_count

        if imbalance_ratio > 100:
            strategy_type = 'severe'
        elif imbalance_ratio > 20:
            strategy_type = 'high'
        elif imbalance_ratio > 5:
            strategy_type = 'medium'
        else:
            strategy_type = 'mild'

        # محاسبه پارامترهای پویا
        sampling_strategy = {}
        k_neighbors_config = {}

        for cls in class_names:
            current_count = value_counts[cls]

            if strategy_type in ['severe', 'high']:
                # برای عدم تعادل شدید: همه به median برسند
                target_count = median_count
            else:
                # برای عدم تعادل متوسط: تعادل نسبی
                target_count = mean_count

            # محدود کردن target_count به محدوده معقول
            target_count = max(min(target_count, majority_count), minority_count)
            sampling_strategy[cls] = target_count

            # محاسبه k_neighbors برای SMOTE
            if current_count < target_count:  # نیاز به oversampling
                k_neighbors = min(5, current_count - 1)
                k_neighbors = max(1, k_neighbors)  # حداقل 1
                k_neighbors_config[cls] = k_neighbors

        return {
            'sampling_strategy': sampling_strategy,
            'k_neighbors_config': k_neighbors_config,
            'strategy_type': strategy_type,
            'imbalance_ratio': imbalance_ratio
        }

    def create_undersampler(self, y_train: pd.Series) -> Tuple[Any, pd.Series]:
        """ایجاد Undersampler پویا"""
        dynamic_params = self.calculate_dynamic_parameters(y_train)
        sampling_strategy = dynamic_params['sampling_strategy']

        # فقط کلاس‌هایی که نیاز به undersampling دارند
        undersample_strategy = {
            cls: target for cls, target in sampling_strategy.items()
            if target < y_train.value_counts()[cls]
        }

        if not undersample_strategy:
            print("⚠️  هیچ کلاسی نیاز به undersampling ندارد")
            return None, y_train

        try:
            undersampler = RandomUnderSampler(
                sampling_strategy=undersample_strategy,
                random_state=self.config.RANDOM_STATE
            )

            # برای تطبیق با signature، باید X و y با هم resample شوند
            # در اینجا فقط strategy را برمی‌گردانیم
            self.samplers['undersampler'] = undersampler
            self.sampling_reports['undersampling'] = {
                'strategy': undersample_strategy,
                'parameters': dynamic_params
            }

            return undersampler, None

        except Exception as e:
            print(f"❌ خطا در ایجاد undersampler: {e}")
            return None, y_train

    def create_oversampler(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, pd.Series]:
        """ایجاد Oversampler پویا با مدیریت خطا"""

        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"⚠️  ستون‌های غیرعددی شناسایی شدند: {list(non_numeric_cols)}")
            # می‌توانید اینجا تصمیم بگیرید که حذف شوند یا encode شوند

        dynamic_params = self.calculate_dynamic_parameters(y_train)
        sampling_strategy = dynamic_params['sampling_strategy']

        # فیلتر کردن کلاس‌های معتبر برای oversampling
        valid_oversample_strategy = {}
        problematic_classes = []

        for cls, target in sampling_strategy.items():
            if cls not in y_train.values:
                continue

            current_count = (y_train == cls).sum()
            if target > current_count and current_count > 0:
                valid_oversample_strategy[cls] = target

                if current_count < self.config.MIN_SAMPLES_FOR_SMOTE:
                    problematic_classes.append(cls)

        if not valid_oversample_strategy:
            print("⚠️  هیچ کلاسی نیاز به oversampling ندارد")
            return None, y_train

        dynamic_params = self.calculate_dynamic_parameters(y_train)
        sampling_strategy = dynamic_params['sampling_strategy']
        k_neighbors_config = dynamic_params['k_neighbors_config']

        # فقط کلاس‌هایی که نیاز به oversampling دارند
        oversample_strategy = {
            cls: target for cls, target in sampling_strategy.items()
            if target > y_train.value_counts()[cls]
        }

        if not oversample_strategy:
            print("⚠️  هیچ کلاسی نیاز به oversampling ندارد")
            return None, y_train

        # بررسی امکان استفاده از SMOTE
        smote_possible = True
        problematic_classes = []

        for cls, target in oversample_strategy.items():
            current_count = y_train.value_counts()[cls]
            if current_count < self.config.MIN_SAMPLES_FOR_SMOTE:
                smote_possible = False
                problematic_classes.append(cls)

        try:
            if smote_possible:
                # استفاده از SMOTE
                oversampler = SMOTE(
                    sampling_strategy=oversample_strategy,
                    random_state=self.config.RANDOM_STATE,
                    k_neighbors=2  # مقدار پایه، در حین اجرا تطبیق داده می‌شود
                )
                method_name = "SMOTE"
            else:
                # استفاده از RandomOverSampler برای کلاس‌های با نمونه کم
                print(f"⚠️  استفاده از RandomOverSampler برای کلاس‌های: {problematic_classes}")
                oversampler = RandomOverSampler(
                    sampling_strategy=oversample_strategy,
                    random_state=self.config.RANDOM_STATE
                )
                method_name = "RandomOverSampler"

            self.samplers['oversampler'] = oversampler
            self.sampling_reports['oversampling'] = {
                'strategy': oversample_strategy,
                'method': method_name,
                'parameters': dynamic_params,
                'problematic_classes': problematic_classes if not smote_possible else []
            }

            return oversampler, None

        except Exception as e:
            print(f"❌ خطا در ایجاد oversampler: {e}")
            # fallback به RandomOverSampler
            try:
                oversampler = RandomOverSampler(
                    sampling_strategy=oversample_strategy,
                    random_state=self.config.RANDOM_STATE
                )
                self.samplers['oversampler'] = oversampler
                return oversampler, None
            except:
                return None, y_train

    def apply_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """اعمال تمام استراتژی‌های نمونه‌برداری"""
        results = {
            'original': {'X': X_train, 'y': y_train}
        }

        # ۱. Undersampling
        undersampler, _ = self.create_undersampler(y_train)
        if undersampler is not None:
            try:
                X_under, y_under = undersampler.fit_resample(X_train, y_train)
                results['undersampled'] = {'X': X_under, 'y': y_under}
                print(f"✅ Undersampling انجام شد: {X_under.shape}")
            except Exception as e:
                print(f"❌ خطا در undersampling: {e}")

        # ۲. Oversampling
        oversampler, _ = self.create_oversampler(X_train, y_train)
        if oversampler is not None:
            try:
                X_over, y_over = oversampler.fit_resample(X_train, y_train)
                results['oversampled'] = {'X': X_over, 'y': y_over}
                print(f"✅ Oversampling انجام شد: {X_over.shape}")
            except Exception as e:
                print(f"❌ خطا در oversampling: {e}")

        return results

    def get_sampling_report(self) -> Dict[str, Any]:
        """گزارش نمونه‌برداری"""
        return self.sampling_reports