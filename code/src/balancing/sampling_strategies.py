import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings

warnings.filterwarnings('ignore')


class AdaptiveSamplingStrategies:
    """استراتژی‌های نمونه‌برداری پویا و هوشمند"""

    def __init__(self, config):
        self.config = config
        self.samplers = {}
        self.sampling_reports = {}

    # ----------------------------------------------------------------------
    def calculate_dynamic_parameters(self, y_train: pd.Series) -> Dict[str, Any]:
        """
        محاسبه مقادیر هدف برای هر کلاس بر اساس توزیع داده به صورت داینامیک.
        این متد مشخص می‌کند کدام کلاس‌ها نیاز به Oversampling یا Undersampling دارند.
        """
        value_counts = y_train.value_counts()
        class_names = value_counts.index.tolist()
        majority_class = value_counts.idxmax()
        majority_count = value_counts.max()

        # میانگین و میانه برای تحلیل مقیاس داده‌ها
        mean_count = int(value_counts.mean())
        median_count = int(value_counts.median())

        # پارامترهای پایه داینامیک
        alpha = 8  # ضریب رشد برای کلاس‌های کم‌نمونه
        gamma = 6  # ضریب کاهش برای کلاس‌های خیلی بزرگ
        target_minority = int(np.sqrt(majority_count) * alpha)
        target_majority = int(target_minority * gamma)

        sampling_strategy_over = {}
        sampling_strategy_under = {}

        # آستانه‌ها برای شناسایی کلاس‌های کوچک و بزرگ
        small_threshold = 0.2 * majority_count  # کمتر از ۲۰٪ بیشترین کلاس
        big_threshold = 0.8 * majority_count   # بیشتر از ۸۰٪ بیشترین کلاس

        for cls, count in value_counts.items():
            if count < small_threshold:
                # کلاس کم‌نمونه → افزایش تا target_minority
                sampling_strategy_over[cls] = target_minority
            elif count > big_threshold:
                # کلاس خیلی بزرگ → کاهش تا target_majority
                sampling_strategy_under[cls] = target_majority

        strategy_type = {
            'oversample_classes': list(sampling_strategy_over.keys()),
            'undersample_classes': list(sampling_strategy_under.keys())
        }

        return {
            'sampling_strategy_over': sampling_strategy_over,
            'sampling_strategy_under': sampling_strategy_under,
            'target_minority': target_minority,
            'target_majority': target_majority,
            'strategy_type': strategy_type
        }

    # ----------------------------------------------------------------------
    def create_oversampler(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, pd.Series]:
        """ایجاد Oversampler فقط برای کلاس‌های کم‌نمونه با SMOTE یا RandomOverSampler"""

        dynamic_params = self.calculate_dynamic_parameters(y_train)
        oversample_strategy = dynamic_params['sampling_strategy_over']

        if not oversample_strategy:
            print("⚠️  هیچ کلاسی نیاز به Oversampling ندارد")
            return None, y_train

        # بررسی حداقل داده برای استفاده از SMOTE
        min_class_size = y_train.value_counts().min()
        use_smote = min_class_size >= self.config.MIN_SAMPLES_FOR_SMOTE

        try:
            if use_smote:
                oversampler = SMOTE(
                    sampling_strategy=oversample_strategy,
                    random_state=self.config.RANDOM_STATE,
                    k_neighbors=3
                )
                method_name = "SMOTE"
            else:
                print("⚠️  داده‌ی برخی کلاس‌ها خیلی کم است، استفاده از RandomOverSampler")
                oversampler = RandomOverSampler(
                    sampling_strategy=oversample_strategy,
                    random_state=self.config.RANDOM_STATE
                )
                method_name = "RandomOverSampler"

            self.samplers['oversampler'] = oversampler
            self.sampling_reports['oversampling'] = {
                'strategy': oversample_strategy,
                'method': method_name,
                'parameters': dynamic_params
            }
            return oversampler, None

        except Exception as e:
            print(f"❌ خطا در ایجاد Oversampler: {e}")
            return None, y_train

    # ----------------------------------------------------------------------
    def create_undersampler(self, y_train: pd.Series) -> Tuple[Any, pd.Series]:
        """ایجاد Undersampler فقط برای کلاس‌های خیلی بزرگ"""

        dynamic_params = self.calculate_dynamic_parameters(y_train)
        undersample_strategy = dynamic_params['sampling_strategy_under']

        if not undersample_strategy:
            print("⚠️  هیچ کلاسی نیاز به Undersampling ندارد")
            return None, y_train

        try:
            undersampler = RandomUnderSampler(
                sampling_strategy=undersample_strategy,
                random_state=self.config.RANDOM_STATE
            )

            self.samplers['undersampler'] = undersampler
            self.sampling_reports['undersampling'] = {
                'strategy': undersample_strategy,
                'parameters': dynamic_params
            }

            return undersampler, None

        except Exception as e:
            print(f"❌ خطا در ایجاد Undersampler: {e}")
            return None, y_train

    # ----------------------------------------------------------------------
    def apply_sampling(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """اعمال استراتژی‌ها و تولید سه دیتاست: baseline, Oversampling, Undersampling"""
        results = {
            'baseline': {'X': X_train, 'y': y_train}
        }

        # === ۱. Oversampling ===
        oversampler, _ = self.create_oversampler(X_train, y_train)
        if oversampler is not None:
            try:
                X_over, y_over = oversampler.fit_resample(X_train, y_train)
                results['oversampling'] = {'X': X_over, 'y': y_over}
                print(f"✅ Oversampling انجام شد → {X_over.shape}")
            except Exception as e:
                print(f"❌ خطا در Oversampling: {e}")

        # === ۲. Undersampling ===
        undersampler, _ = self.create_undersampler(y_train)
        if undersampler is not None:
            try:
                X_under, y_under = undersampler.fit_resample(X_train, y_train)
                results['undersampling'] = {'X': X_under, 'y': y_under}
                print(f"✅ Undersampling انجام شد → {X_under.shape}")
            except Exception as e:
                print(f"❌ خطا در Undersampling: {e}")

        return results

    # ----------------------------------------------------------------------
    def get_sampling_report(self) -> Dict[str, Any]:
        """گزارش نمونه‌برداری"""
        return self.sampling_reports
