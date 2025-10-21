from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
from datetime import datetime

# اضافه کردن مسیر ماژول‌ها
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_manager'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from model_manager.model_trainer import ModelTrainer
from model_manager.model_evaluator import ModelEvaluator
from config.model_config import Phase4Config


class Phase4ModelTraining:
    """کلاس اصلی فاز ۴ - مدیریت کامل مدل‌سازی و ارزیابی"""

    def __init__(self, config=None):
        self.config = config or Phase4Config()
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.results = {}

    def run_training_pipeline(self, data_dir: str = "data/balancedData") -> Dict[str, Any]:
        """اجرای کامل پایپلاین فاز ۴"""
        print("=" * 60)
        print("فاز ۴: مدل‌سازی و ارزیابی")
        print("=" * 60)

        data_path = Path(data_dir)

        try:
            # ۱. بارگذاری داده‌های Train
            print("📥 مرحله ۱: بارگذاری داده‌های آموزش...")
            train_strategies = self._load_train_data(data_path)

            # ۲. بارگذاری داده‌های Test
            print("\n📊 مرحله ۲: بارگذاری داده‌های تست...")
            X_test, y_test = self._load_test_data(data_path)

            # ۳. آموزش مدل‌های پایه
            print("\n🎯 مرحله ۳: آموزش مدل‌های پایه...")
            base_models = {}
            for strategy_name, (X_train, y_train) in train_strategies.items():
                base_models[strategy_name] = self.trainer.train_base_models(X_train, y_train, strategy_name)

            print("\n💾 مرحله ۵: ذخیره‌سازی مدل‌های پایه برای همه استراتژی‌ها...")
            for strategy_name in train_strategies.keys():
                self.trainer.save_models(base_models, {}, strategy_name)

            # ۴. تنظیم هایپرپارامترها (روی oversampled)
            print("\n⚙️  مرحله ۴: تنظیم هایپرپارامترها...")
            if 'oversampled' in train_strategies:
                X_train_over, y_train_over = train_strategies['oversampled']
                tuned_models, tuning_results = self.trainer.hyperparameter_tuning(
                    X_train_over, y_train_over, 'oversampled'
                )
                self.results['tuning_results'] = tuning_results

                #  ذخیره‌سازی مدل‌های تنظیم‌شده
                print("\n💾 ذخیره‌سازی مدل‌های تنظیم‌شده...")
                self.trainer.save_models({}, tuned_models, 'oversampled')
            else:
                tuned_models = {}
                print("   ⚠️  داده oversampled برای تنظیم یافت نشد")


            # ۵. ذخیره‌سازی مدل‌ها
            print("\n💾 مرحله ۵: ذخیره‌سازی مدل‌ها...")
            self.trainer.save_models(base_models, tuned_models, 'oversampled')

            # ۶. ارزیابی مدل‌ها
            print("\n📈 مرحله ۶: ارزیابی مدل‌ها روی تست set...")
            evaluation_results = {}
            for strategy_name in train_strategies.keys():
                strategy_eval = self.evaluator.evaluate_models(X_test, y_test, strategy_name)
                evaluation_results[strategy_name] = strategy_eval

            self.results['evaluation_results'] = evaluation_results

            # ۷. تولید گزارش‌ها
            print("\n📋 مرحله ۷: تولید گزارش‌های نهایی...")
            output_dir = Path(self.config.MODEL_SAVE_PATHS['evaluation'])
            self.evaluator.generate_reports(output_dir)

            # ۸. انتخاب مدل برتر
            print("\n🏆 مرحله ۸: انتخاب مدل برتر...")
            best_model_info = self._select_best_model(evaluation_results)
            self.results['best_model'] = best_model_info

            print(f"\n🎉 فاز ۴ با موفقیت completed شد!")
            print(f"📁 نتایج در: {output_dir}")

            return self.results

        except Exception as e:
            print(f"❌ خطا در اجرای فاز ۴: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_train_data(self, data_path: Path) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """بارگذاری داده‌های آموزش از استراتژی‌های مختلف"""
        strategies = ['original', 'undersampled', 'oversampled']
        train_data = {}

        for strategy in strategies:
            strategy_path = data_path / strategy
            X_path = strategy_path / "X_train.csv"
            y_path = strategy_path / "y_train.csv"

            if X_path.exists() and y_path.exists():
                X_train = pd.read_csv(X_path)
                y_train = pd.read_csv(y_path).iloc[:, 0]  # اولین ستون هدف
                train_data[strategy] = (X_train, y_train)
                print(f"   ✅ {strategy}: {X_train.shape}")
            else:
                print(f"   ⚠️  داده {strategy} یافت نشد")

        return train_data

    def _load_test_data(self, data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """بارگذاری داده‌های تست"""
        test_path = data_path / "test"
        X_test = pd.read_csv(test_path / "X_test.csv")
        y_test = pd.read_csv(test_path / "y_test.csv").iloc[:, 0]

        print(f"   ✅ Test set: {X_test.shape}")
        return X_test, y_test

    def _select_best_model(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """انتخاب بهترین مدل بر اساس معیارهای امنیتی"""
        best_model = None
        best_score = -1
        best_details = {}

        for strategy_name, models_eval in evaluation_results.items():
            for model_name, eval_result in models_eval.items():
                if 'security_metrics' in eval_result:
                    security_score = (
                            eval_result['security_metrics']['mean_security_recall'] * 0.4 +
                            eval_result['security_metrics']['security_f1'] * 0.3 +
                            eval_result['security_metrics']['threat_detection_rate'] * 0.3
                    )

                    if security_score > best_score:
                        best_score = security_score
                        best_model = f"{strategy_name}_{model_name}"
                        best_details = {
                            'strategy': strategy_name,
                            'model': model_name,
                            'security_score': security_score,
                            'security_recall': eval_result['security_metrics']['mean_security_recall'],
                            'security_f1': eval_result['security_metrics']['security_f1'],
                            'threat_detection_rate': eval_result['security_metrics']['threat_detection_rate'],
                            'accuracy': eval_result['general_metrics']['accuracy'],
                            'f1_macro': eval_result['general_metrics']['f1_macro']
                        }

        print(f"   🏆 بهترین مدل: {best_model}")
        print(f"   📊 امتیاز امنیتی: {best_score:.3f}")
        print(f"   🛡️  Recall امنیتی: {best_details['security_recall']:.3f}")

        return best_details


def main():
    """تابع اصلی اجرای فاز ۴"""
    print("🚀 اجرای فاز ۴: مدل‌سازی و ارزیابی")

    try:
        # ایجاد مدل‌ساز
        model_training = Phase4ModelTraining()

        # اجرای پایپلاین
        results = model_training.run_training_pipeline()

        if results:
            print(f"\n📊 خلاصه نتایج فاز ۴:")
            print(f"   • استراتژی‌های آموزش دیده: {list(results.get('evaluation_results', {}).keys())}")

            best_model = results.get('best_model', {})
            if best_model:
                print(f"   • بهترین مدل: {best_model['strategy']}_{best_model['model']}")
                print(f"   • امتیاز امنیتی: {best_model['security_score']:.3f}")
                print(f"   • Recall امنیتی: {best_model['security_recall']:.3f}")

            return results
        else:
            print("❌ فاز ۴ با خطا مواجه شد")
            return None

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۴: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()