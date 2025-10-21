import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# اضافه کردن مسیر ماژول‌ها
sys.path.append(os.path.join(os.path.dirname(__file__)))

from feature_engineers.smart_feature_engineer import SmartFeatureEngineer


class FeatureEngineeringPipeline:
    """پایپلاین اصلی مهندسی ویژگی - عمومی و قابل استفاده مجدد"""

    def __init__(self, metadata_path, scaling_strategy='standard'):
        self.metadata_path = metadata_path
        self.scaling_strategy = scaling_strategy
        self.engineer = None
        self.is_fitted = False

    def fit_transform(self, df, target_column='Action'):
        """آموزش و تبدیل داده‌ها"""
        print("🚀 شروع فرآیند مهندسی ویژگی‌ها")
        print("=" * 60)
        print(f"🔍 بررسی ستون هدف '{target_column}':")
        print(f"   نوع داده: {df[target_column].dtype}")
        print(f"   مقادیر منحصر به فرد: {df[target_column].unique()}")

        # ایجاد مهندس ویژگی
        self.engineer = SmartFeatureEngineer(
            metadata_path=self.metadata_path,
            target_column=target_column,
            scaling_strategy=self.scaling_strategy
        )

        # اجرای مهندسی ویژگی
        df_engineered = self.engineer.fit_transform(df)
        self.is_fitted = True

        print(f"🔍 ستون هدف پس از پردازش:")
        print(f"   نوع داده: {df_engineered[target_column].dtype}")
        print(f"   مقادیر منحصر به فرد: {df_engineered[target_column].unique()}")

        # ذخیره‌سازی نتایج
        self._save_results(df_engineered)

        return df_engineered

    def transform(self, df):
        """تبدیل داده‌های جدید"""
        if not self.is_fitted or self.engineer is None:
            raise Exception("ابتدا باید متد fit_transform فراخوانی شود")

        return self.engineer.transform(df)

    def _convert_to_serializable(self, obj):
        """تبدیل انواع غیرقابل سریالایز به انواع استاندارد پایتون"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _save_results(self, df_final):
        """ذخیره‌سازی نتایج و گزارش"""
        print("\n💾 ذخیره‌سازی نتایج...")

        # ایجاد پوشه خروجی
        output_dir = Path(__file__).parent.parent / "data/engineeredData"
        output_dir.mkdir(exist_ok=True)

        # ذخیره dataset نهایی
        output_path = output_dir / "engineered_dataset.csv"
        df_final.to_csv(output_path, index=False)
        print(f"   ✅ dataset نهایی ذخیره شد: {output_path}")
        print(f"مقادیر null: {df_final.isnull().sum().sum()}")

        # ایجاد گزارش
        report = self._generate_engineering_report(df_final)
        report_path = output_dir / "feature_engineering_report.json"

        # تبدیل به انواع قابل سریالایز
        serializable_report = self._convert_to_serializable(report)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        print(f"   ✅ گزارش مهندسی ویژگی ذخیره شد: {report_path}")

        # ذخیره pipeline (برای استفاده آینده)
        pipeline_info = {
            'metadata_path': str(self.metadata_path),
            'scaling_strategy': self.scaling_strategy,
            'fitted': self.is_fitted,
            'feature_summary': self.engineer.get_feature_summary() if self.engineer else None
        }

        # تبدیل pipeline info به انواع قابل سریالایز
        serializable_pipeline_info = self._convert_to_serializable(pipeline_info)

        pipeline_path = output_dir / "feature_pipeline_info.json"
        with open(pipeline_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_pipeline_info, f, indent=2, ensure_ascii=False)

        print(f"   ✅ اطلاعات pipeline ذخیره شد: {pipeline_path}")

        return output_path, report_path

    def _generate_engineering_report(self, df_final):
        """تولید گزارش مهندسی ویژگی"""
        if self.engineer is None:
            return {}

        feature_summary = self.engineer.get_feature_summary()

        # اطمینان از قابل سریالایز بودن توزیع هدف
        target_dist = {}
        if self.engineer.target_column in df_final.columns:
            target_counts = df_final[self.engineer.target_column].value_counts()
            target_dist = {str(k): int(v) for k, v in target_counts.items()}

        report = {
            "metadata": {
                "phase": "Feature Engineering - Phase 2",
                "timestamp": pd.Timestamp.now().isoformat(),
                "final_shape": {
                    "rows": int(df_final.shape[0]),
                    "columns": int(df_final.shape[1])
                },
                "scaling_strategy": self.scaling_strategy,
                "target_column": self.engineer.target_column,
                "is_fitted": self.is_fitted
            },
            "feature_analysis": {
                "total_features": int(len(df_final.columns)),
                "numeric_features": int(len(df_final.select_dtypes(include=[np.number]).columns)),
                "categorical_features": int(len(df_final.select_dtypes(include=['object', 'category']).columns)),
                "target_distribution": target_dist
            },
            "engineer_summary": feature_summary,
            "data_quality": {
                "missing_values": int(df_final.isnull().sum().sum()),
                "memory_usage_mb": float(round(df_final.memory_usage(deep=True).sum() / 1024 / 1024, 2))
            },
            "process_summary": {
                "nat_features_created": len([col for col in df_final.columns if 'nat' in col.lower()]),
                "port_features_created": len([col for col in df_final.columns if 'port' in col.lower()]),
                "traffic_features_created": len(
                    [col for col in df_final.columns if any(x in col.lower() for x in ['byte', 'packet', 'ratio'])]),
                "time_features_created": len(
                    [col for col in df_final.columns if any(x in col.lower() for x in ['time', 'duration', 'second'])]),
                "log_features_created": len([col for col in df_final.columns if 'log1p' in col.lower()])
            }
        }

        return report


# تابع اصلی برای اجرای مستقل
def main(input_file=None, metadata_file=None, scaling_strategy='standard'):
    """تابع اصلی اجرای فاز ۲"""
    if input_file is None:
        input_file = "../data/cleaned_network_data.csv"
    if metadata_file is None:
        metadata_file = "../data/columns_metadata.json"

    print("=" * 60)
    print("فاز ۲: مهندسی ویژگی هوشمند و عمومی")
    print("=" * 60)

    try:
        # بارگذاری داده‌های پاک‌شده
        df = pd.read_csv(input_file)
        print(f"📁 داده‌های پاک‌شده بارگذاری شد: {df.shape}")
        print(f"مقادیر null: {df.isnull().sum().sum()}")

        # ایجاد و اجرای پایپلاین
        pipeline = FeatureEngineeringPipeline(
            metadata_path=metadata_file,
            scaling_strategy=scaling_strategy
        )

        df_engineered = pipeline.fit_transform(df)

        print(f"\n🎉 فاز ۲ completed شد!")
        print(f"📊 ابعاد نهایی: {df_engineered.shape}")

        # نمایش خلاصه‌ای از ویژگی‌های ایجاد شده
        print(f"\n📈 خلاصه ویژگی‌های ایجاد شده:")
        print(f"   • کل ویژگی‌ها: {len(df_engineered.columns)}")
        print(f"   • ویژگی‌های عددی: {len(df_engineered.select_dtypes(include=[np.number]).columns)}")
        print(f"   • ویژگی‌های دسته‌ای: {len(df_engineered.select_dtypes(include=['object', 'category']).columns)}")

        # نمایش توزیع هدف
        if 'Action' in df_engineered.columns:
            target_dist = df_engineered['Action'].value_counts()
            print(f"   • توزیع کلاس هدف:")
            for cls, count in target_dist.items():
                print(f"     {cls}: {count} ({count / len(df_engineered) * 100:.1f}%)")

        return df_engineered, pipeline

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۲: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='فاز ۲: مهندسی ویژگی هوشمند')
    parser.add_argument('--input', type=str, help='فایل ورودی', default='../data/cleanedData/network_logs_cleaned.csv')
    parser.add_argument('--metadata', type=str, help='فایل متادیتا', default='../data/columns_metadata.json')
    parser.add_argument('--scaling', choices=['standard', 'robust'],
                        default='standard', help='استراتژی مقیاس‌بندی')

    args = parser.parse_args()

    main(input_file=args.input, metadata_file=args.metadata, scaling_strategy=args.scaling)