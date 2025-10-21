import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """مدیر آموزش مدل‌های طبقه‌بندی - عمومی و قابل استفاده مجدد"""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.training_reports = {}

    def create_preprocessor(self, strategy_name: str) -> Pipeline:
        """ایجاد پایپلاین پیش‌پردازش برای استراتژی خاص"""
        preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])
        return preprocessor

    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                          strategy_name: str) -> Dict[str, Any]:
        """آموزش تمام مدل‌های پایه برای یک استراتژی"""
        print(f"🎯 آموزش مدل‌های پایه برای استراتژی: {strategy_name}")

        # ایجاد preprocessor
        preprocessor = self.create_preprocessor(strategy_name)

        # آموزش preprocessor
        X_processed = preprocessor.fit_transform(X_train)
        self.preprocessors[strategy_name] = preprocessor

        trained_models = {}
        training_results = {}

        for model_name, model_config in self.config.BASE_MODELS.items():
            print(f"   🔧 آموزش {model_name}...")

            try:
                # ایجاد مدل
                model_class = globals()[model_config['class']]
                model = model_class(**model_config['params'])

                # آموزش مدل
                model.fit(X_processed, y_train)

                # ذخیره مدل
                trained_models[model_name] = model
                training_results[model_name] = {
                    'status': 'success',
                    'training_samples': len(X_train),
                    'feature_count': X_processed.shape[1]
                }

                print(f"   ✅ {model_name} آموزش داده شد")

            except Exception as e:
                training_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"   ❌ خطا در آموزش {model_name}: {e}")

        self.models[strategy_name] = trained_models
        self.training_reports[strategy_name] = training_results

        return trained_models

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                              strategy_name: str, model_names: List[str] = None) -> Dict[str, Any]:
        """تنظیم هایپرپارامترها برای مدل‌های انتخابی - اصلاح نهایی"""
        if model_names is None:
            model_names = ['svm', 'random_forest']

        print(f"⚙️  تنظیم هایپرپارامترها برای استراتژی: {strategy_name}")

        tuned_models = {}
        tuning_results = {}

        # استفاده از preprocessor آموزش دیده
        preprocessor = self.preprocessors.get(strategy_name)
        if preprocessor is None:
            print(f"   ⚠️  Preprocessor برای {strategy_name} یافت نشد - ایجاد جدید")
            preprocessor = self.create_preprocessor(strategy_name)
            X_processed = preprocessor.fit_transform(X_train)
            self.preprocessors[strategy_name] = preprocessor
        else:
            X_processed = preprocessor.transform(X_train)

        # 🔧 اصلاح: استفاده از RandomState برای نمونه‌برداری
        if len(X_processed) > 5000:
            print(f"   📉 کاهش حجم داده از {len(X_processed)} به 5000 نمونه برای تنظیم‌هایپرپارامتر")

            # 🔥 اصلاح خطا: استفاده از RandomState به جای random_state
            rng = np.random.RandomState(self.config.RANDOM_STATE)
            indices = rng.choice(len(X_processed), size=5000, replace=False)

            X_sampled = X_processed[indices]
            y_sampled = y_train.iloc[indices]
        else:
            X_sampled = X_processed
            y_sampled = y_train

        print(f"   ✅ داده‌های نمونه‌گیری شده: X={X_sampled.shape}, y={len(y_sampled)}")

        for model_name in model_names:
            if model_name not in self.config.HYPERPARAM_GRIDS:
                print(f"   ⚠️  هایپرپارامترهای {model_name} تعریف نشده")
                continue

            print(f"   🔍 تنظیم {model_name}...")

            try:
                # ایجاد مدل پایه
                model_config = self.config.BASE_MODELS[model_name]
                model_class = globals()[model_config['class']]
                base_model = model_class(**model_config['params'])

                # جستجوی هایپرپارامتر
                param_grid = self.config.HYPERPARAM_GRIDS[model_name]

                # استفاده از RandomizedSearchCV
                search = RandomizedSearchCV(
                    base_model, param_grid,
                    n_iter=10,
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=self.config.N_JOBS,
                    random_state=self.config.RANDOM_STATE,
                    verbose=1
                )

                # اجرای جستجو روی داده‌های نمونه‌گیری شده
                print(f"   ⏳ شروع جستجوی هایپرپارامتر برای {model_name}...")
                search.fit(X_sampled, y_sampled)

                # آموزش مدل نهایی روی تمام داده‌ها با بهترین پارامترها
                print(f"   🎯 آموزش مدل نهایی {model_name} روی تمام داده‌ها...")
                best_model = model_class(**search.best_params_)
                best_model.fit(X_processed, y_train)

                # ذخیره مدل تنظیم‌شده
                tuned_models[model_name] = best_model
                tuning_results[model_name] = {
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'training_samples': len(X_processed),
                    'status': 'success'
                }

                print(f"   ✅ {model_name} تنظیم شد (best score: {search.best_score_:.3f})")

            except Exception as e:
                tuning_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"   ❌ خطا در تنظیم {model_name}: {e}")

        return tuned_models, tuning_results


    def _simplify_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """ساده‌سازی نتایج CV برای ذخیره‌سازی"""
        simplified = {}
        for key, values in cv_results.items():
            if key.startswith('param_') or key in ['mean_test_score', 'std_test_score', 'rank_test_score']:
                # فقط ۱۰ نتیجه برتر را نگه دار
                if len(values) > 10:
                    simplified[key] = values[:10].tolist() if hasattr(values, 'tolist') else values[:10]
                else:
                    simplified[key] = values.tolist() if hasattr(values, 'tolist') else values
        return simplified

    def save_models(self, base_models: Dict[str, Dict[str, Any]],
                    tuned_models: Dict[str, Any], strategy_name: str):
        """ذخیره‌سازی تمام مدل‌ها و preprocessorها"""

        # ایجاد پوشه‌ها
        base_path = Path(self.config.MODEL_SAVE_PATHS['base_models']) / strategy_name
        tuned_path = Path(self.config.MODEL_SAVE_PATHS['tuned_models'])
        preprocessor_path = Path(self.config.MODEL_SAVE_PATHS['preprocessors'])

        base_path.mkdir(parents=True, exist_ok=True)
        tuned_path.mkdir(parents=True, exist_ok=True)
        preprocessor_path.mkdir(parents=True, exist_ok=True)

        # ذخیره مدل‌های پایه
        for model_name, model in base_models.get(strategy_name, {}).items():
            model_path = base_path / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"   💾 مدل پایه {model_name} ذخیره شد")

        # ذخیره مدل‌های تنظیم‌شده
        for model_name, model in tuned_models.items():
            model_path = tuned_path / f"{model_name}_{strategy_name}_tuned.pkl"
            joblib.dump(model, model_path)
            print(f"   💾 مدل تنظیم‌شده {model_name} ذخیره شد")

        # ذخیره preprocessor
        preprocessor = self.preprocessors.get(strategy_name)
        if preprocessor:
            preprocessor_path_file = preprocessor_path / f"{strategy_name}_preprocessor.pkl"
            joblib.dump(preprocessor, preprocessor_path_file)
            print(f"   💾 Preprocessor {strategy_name} ذخیره شد")
        else:
            print(f"   ⚠️  Preprocessor برای {strategy_name} یافت نشد")

        print(f"💾 مدل‌های استراتژی {strategy_name} ذخیره شدند")

    def get_training_summary(self) -> Dict[str, Any]:
        """خلاصه نتایج آموزش"""
        summary = {
            'total_strategies': len(self.models),
            'strategies_trained': list(self.models.keys()),
            'models_per_strategy': {},
            'preprocessors_trained': list(self.preprocessors.keys())
        }

        for strategy, models in self.models.items():
            summary['models_per_strategy'][strategy] = {
                'total_models': len(models),
                'successful_models': sum(1 for m in models.values() if m is not None),
                'model_names': list(models.keys())
            }

        return summary