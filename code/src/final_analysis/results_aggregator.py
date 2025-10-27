import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class ResultsAggregator:
    """جمع‌آوری و خلاصه‌سازی نتایج مدل‌ها"""

    def __init__(self, config):
        self.config = config
        self.model_summary = pd.DataFrame()
        self.detailed_results = {}

    def load_phase4_results(self, evaluation_path: Path) -> bool:
        """بارگذاری نتایج فاز ۴ و نرمال‌سازی نام استراتژی‌ها"""
        try:
            # ۱. بارگذاری گزارش مقایسه‌ای کلی
            comp_path = evaluation_path / "comparative_analysis.json"
            with open(comp_path, 'r', encoding='utf-8') as f:
                comparative_data = json.load(f)

            # ۲. نگاشت نام‌های قدیمی به جدید
            name_map = {
                'original': 'baseline',
                'oversampled': 'oversampling',
                'undersampled': 'undersampling',
                'oversampling': 'oversampling',
                'undersampling': 'undersampling',
                'baseline': 'baseline'
            }

            # ۳. بارگذاری گزارش‌های تفصیلی
            detailed_data = {}
            for raw_name in ['original', 'undersampled', 'oversampled',
                             'baseline', 'undersampling', 'oversampling']:
                strategy = name_map.get(raw_name, raw_name)
                detail_path = evaluation_path / f"detailed_report_{raw_name}.json"
                if not detail_path.exists():
                    # بررسی اگر فایل با نام نرمال شده وجود دارد
                    alt_path = evaluation_path / f"detailed_report_{strategy}.json"
                    if alt_path.exists():
                        detail_path = alt_path
                    else:
                        continue
                with open(detail_path, 'r', encoding='utf-8') as f:
                    detailed_data[strategy] = json.load(f)

            # ۴. ذخیره داده‌ها در حافظه داخلی
            self.detailed_results = {
                'comparative': comparative_data,
                'detailed': detailed_data
            }

            print(f"✅ نتایج فاز ۴ بارگذاری شدند ({len(detailed_data)} استراتژی یافت شد)")
            return True

        except Exception as e:
            print(f"❌ خطا در بارگذاری نتایج فاز ۴: {e}")
            return False

    def create_model_summary_table(self) -> pd.DataFrame:
        """ایجاد جدول خلاصه مدل‌ها"""
        summary_data = []

        for strategy, models_data in self.detailed_results['detailed'].items():
            for model_name, model_results in models_data.items():
                if 'general_metrics' in model_results:
                    row = self._extract_model_metrics(strategy, model_name, model_results)
                    summary_data.append(row)

        self.model_summary = pd.DataFrame(summary_data)

        # محاسبه معیارهای ترکیبی
        self._calculate_composite_metrics()

        return self.model_summary

    def _extract_model_metrics(self, strategy: str, model_name: str,
                               model_results: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج معیارهای هر مدل"""
        general_metrics = model_results['general_metrics']
        security_metrics = model_results['security_metrics']

        row = {
            'model': model_name,
            'dataset': strategy,
            'accuracy': general_metrics['accuracy'],
            'f1_macro': general_metrics['f1_macro'],
            'f1_weighted': general_metrics['f1_weighted'],
            'precision_macro': general_metrics['precision_macro'],
            'recall_macro': general_metrics['recall_macro'],
            'inference_time_ms': 0,  # Placeholder - needs actual measurement
            'model_size_mb': 0,  # Placeholder - needs actual measurement
        }

        # اضافه کردن معیارهای کلاس‌های امنیتی
        for cls in self.config.MINORITY_CLASSES:
            row[f'f1_class_{cls}'] = general_metrics.get(f'f1_class_{cls}', 0)
            row[f'recall_class_{cls}'] = general_metrics.get(f'recall_class_{cls}', 0)
            row[f'precision_class_{cls}'] = general_metrics.get(f'precision_class_{cls}', 0)

        # اضافه کردن معیارهای امنیتی ترکیبی
        row['mean_security_recall'] = security_metrics['mean_security_recall']
        row['security_f1'] = security_metrics['security_f1']
        row['threat_detection_rate'] = security_metrics['threat_detection_rate']

        return row

    def _calculate_composite_metrics(self):
        """محاسبه معیارهای ترکیبی"""
        # میانگین F1 کلاس‌های اقلیت
        f1_minority_cols = [f'f1_class_{cls}' for cls in self.config.MINORITY_CLASSES]
        self.model_summary['f1_minority_mean'] = self.model_summary[f1_minority_cols].mean(axis=1)

        # میانگین Recall کلاس‌های اقلیت
        recall_minority_cols = [f'recall_class_{cls}' for cls in self.config.MINORITY_CLASSES]
        self.model_summary['recall_minority_mean'] = self.model_summary[recall_minority_cols].mean(axis=1)

        # امتیاز امنیتی ترکیبی
        self.model_summary['security_score'] = (
                self.model_summary['f1_minority_mean'] * 0.4 +
                self.model_summary['recall_minority_mean'] * 0.4 +
                self.model_summary['threat_detection_rate'] * 0.2
        )

    def save_summary_tables(self, output_dir: Path):
        """ذخیره جداول خلاصه"""
        # ذخیره CSV
        csv_path = output_dir / "model_summary.csv"
        self.model_summary.to_csv(csv_path, index=False, encoding='utf-8')

        # ذخیره JSON
        json_path = output_dir / "model_summary.json"
        summary_dict = self.model_summary.to_dict('records')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)

        print(f"📊 جداول خلاصه در {output_dir} ذخیره شدند")

        unified_csv = output_dir / "final_summary.csv"
        self.model_summary.to_csv(unified_csv, index=False, encoding='utf-8')
        print(f"📄 جدول نهایی مدل‌ها ذخیره شد: {unified_csv.name}")

    def get_top_models(self, n: int = 5) -> pd.DataFrame:
        """دریافت برترین مدل‌ها بر اساس امتیاز امنیتی"""
        return self.model_summary.nlargest(n, 'security_score')