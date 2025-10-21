
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
from sklearn.model_selection import train_test_split
from typing import Dict, Any

current_dir = os.path.dirname(__file__)
balance_manager_path = os.path.join(current_dir, 'balance_manager')
if balance_manager_path not in sys.path:
    sys.path.insert(0, balance_manager_path)

try:
    from balancing.imbalance_analyzer import ImbalanceAnalyzer
    from balancing.sampling_strategies import AdaptiveSamplingStrategies
    from balancing.data_validator import DataValidator
    from balancing.balance_reporter import BalanceReporter
    from config.balancing_config import BalanceConfig

    print("✅ تمام ماژول‌ها با موفقیت import شدند")
except ImportError as e:
    print(f"❌ خطا در import: {e}")
    sys.exit(1)


class DataBalancer:
    """مدیر هوشمند عدم تعادل کلاس‌ها"""

    def __init__(self, config=None):
        self.config = config or BalanceConfig()
        self.analyzer = ImbalanceAnalyzer(self.config)
        self.sampler = AdaptiveSamplingStrategies(self.config)
        self.validator = DataValidator(self.config)
        self.reporter = BalanceReporter(self.config)

        self.results = {}
        self.reports = {}

    def run_balancing_pipeline(self, input_file: str, output_dir: str = None) -> Dict[str, Any]:
        """اجرای کامل پایپلاین مدیریت عدم تعادل"""
        print("=" * 60)
        print("فاز ۳: مدیریت عدم تعادل کلاس‌ها")
        print("=" * 60)

        # ایجاد پوشه خروجی
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "balancedData"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ۱. بارگذاری داده‌ها
            print("📥 مرحله ۱: بارگذاری داده‌های مهندسی‌شده...")
            df = pd.read_csv(input_file)
            print(f"   ✅ داده‌ها بارگذاری شد: {df.shape}")

            # ۲. تشخیص و مدیریت ستون هدف
            print("\n🎯 مرحله ۲: مدیریت ستون هدف و کلاس‌های نادر...")
            target_column = self.analyzer.detect_target_column(df)
            print(f"   ✅ ستون هدف شناسایی شد: {target_column}")

            # استخراج X و y
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # مدیریت کلاس‌های نادر
            y_processed = self.analyzer.handle_rare_classes(y)

            # ۳. تقسیم داده به Train/Test
            print("\n✂️  مرحله ۳: تقسیم داده به Train/Test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_processed,
                test_size=self.config.TEST_SIZE,
                stratify=y_processed,
                random_state=self.config.RANDOM_STATE
            )

            print(f"   ✅ Train set: {X_train.shape}, Test set: {X_test.shape}")

            # ۴. تحلیل عدم تعادل
            print("\n📊 مرحله ۴: تحلیل عدم تعادل...")
            analysis_report = self.analyzer.analyze_class_distribution(y_train)
            self.reports['analysis'] = analysis_report

            print(f"   ✅ نسبت عدم تعادل: {analysis_report['imbalance_metrics']['imbalance_ratio']:.1f}")
            print(f"   🔧 توصیه‌ها: {len(analysis_report['recommendations'])} مورد")

            # ۵. ایجاد استراتژی‌های نمونه‌برداری
            print("\n🔧 مرحله ۵: ایجاد استراتژی‌های نمونه‌برداری...")
            balanced_data = self.sampler.apply_sampling(X_train, y_train)
            sampling_report = self.sampler.get_sampling_report()
            self.reports['sampling'] = sampling_report

            print(f"   ✅ استراتژی‌های ایجاد شده: {list(balanced_data.keys())}")

            # ۶. اعتبارسنجی
            print("\n🔍 مرحله ۶: اعتبارسنجی داده‌های متعادل...")
            original_data = {'X': X_train, 'y': y_train}
            validation_report = self.validator.validate_balanced_data(original_data, balanced_data)
            self.reports['validation'] = validation_report

            # ۷. ذخیره‌سازی نتایج
            print("\n💾 مرحله ۷: ذخیره‌سازی نتایج...")
            self._save_balanced_datasets(balanced_data, X_test, y_test, output_dir)

            # ۸. تولید گزارش
            print("\n📈 مرحله ۸: تولید گزارش‌ها...")
            final_report = self.reporter.generate_comprehensive_report(
                analysis_report, sampling_report, validation_report, output_dir
            )

            # ۹. مصورسازی
            print("\n🎨 مرحله ۹: ایجاد مصورسازی‌ها...")
            self.analyzer.create_visualizations(
                y_train,
                {k: v['y'] for k, v in balanced_data.items() if k != 'original'},
                output_dir / "balance_visualizations.png"
            )

            print(f"\n🎉 فاز ۳ با موفقیت completed شد!")
            print(f"📁 نتایج در: {output_dir}")

            return {
                'balanced_data': balanced_data,
                'test_data': {'X': X_test, 'y': y_test},
                'reports': self.reports,
                'output_dir': output_dir
            }

        except Exception as e:
            print(f"❌ خطا در اجرای فاز ۳: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _save_balanced_datasets(self, balanced_data: Dict[str, Any],
                                X_test: pd.DataFrame, y_test: pd.Series,
                                output_dir: Path):
        """ذخیره‌سازی تمام مجموعه‌های داده"""

        # ذخیره تست set
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)

        X_test.to_csv(test_dir / "X_test.csv", index=False)
        y_test.to_csv(test_dir / "y_test.csv", index=False)

        # ذخیره هر استراتژی
        for strategy_name, data_dict in balanced_data.items():
            strategy_dir = output_dir / strategy_name
            strategy_dir.mkdir(exist_ok=True)

            X_data = data_dict['X']
            y_data = data_dict['y']

            X_data.to_csv(strategy_dir / "X_train.csv", index=False)
            y_data.to_csv(strategy_dir / "y_train.csv", index=False)

            print(f"   💾 {strategy_name}: {X_data.shape}")

        print(f"   ✅ تمام مجموعه‌های داده ذخیره شدند")


def main(input_file=None, output_dir=None):
    """تابع اصلی اجرای فاز ۳"""
    if input_file is None:
        input_file = "../data/engineered_dataset.csv"

    print("=" * 60)
    print("فاز ۳: مدیریت عدم تعادل کلاس‌ها - اجرای مستقل")
    print("=" * 60)

    try:
        # ایجاد مدیر تعادل
        balance_manager = DataBalancer()

        # اجرای پایپلاین
        results = balance_manager.run_balancing_pipeline(input_file, output_dir)

        if results:
            print(f"\n📊 خلاصه نتایج:")
            print(f"   • استراتژی‌های ایجاد شده: {list(results['balanced_data'].keys())}")
            print(f"   • ابعاد تست set: {results['test_data']['X'].shape}")
            print(f"   • پوشه خروجی: {results['output_dir']}")

            return results
        else:
            print("❌ فاز ۳ با خطا مواجه شد")
            return None

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۳: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='فاز ۳: مدیریت عدم تعادل کلاس‌ها')
    parser.add_argument('--input', type=str, help='فایل ورودی', default='../data/engineeredData/engineered_dataset.csv')
    parser.add_argument('--output', type=str, help='پوشه خروجی', default=None)

    args = parser.parse_args()

    main(input_file=args.input, output_dir=args.output)