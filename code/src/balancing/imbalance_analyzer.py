import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json


class ImbalanceAnalyzer:
    """تحلیل‌گر هوشمند عدم تعادل کلاس‌ها - عمومی برای هر دیتاست"""

    def __init__(self, config):
        self.config = config
        self.analysis_report = {}

    def detect_target_column(self, df: pd.DataFrame) -> str:
        """تشخیص خودکار ستون هدف"""
        for col_name in self.config.TARGET_COLUMN_NAMES:
            if col_name in df.columns:
                return col_name

        # اگر ستون هدف پیدا نشد، آخرین ستون را در نظر بگیر
        return df.columns[-1]

    def analyze_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """تحلیل توزیع کلاس‌ها"""
        value_counts = y.value_counts()
        value_counts_pct = y.value_counts(normalize=True) * 100

        # شناسایی کلاس‌های نادر
        rare_classes = value_counts_pct[value_counts_pct < self.config.RARE_CLASS_THRESHOLD * 100].index.tolist()

        # محاسبه معیارهای عدم تعادل
        imbalance_ratio = value_counts.max() / value_counts.min()
        gini_coefficient = self._calculate_gini_coefficient(value_counts)

        analysis = {
            'class_counts': value_counts.to_dict(),
            'class_percentages': value_counts_pct.round(2).to_dict(),
            'rare_classes': rare_classes,
            'imbalance_metrics': {
                'imbalance_ratio': round(imbalance_ratio, 2),
                'gini_coefficient': round(gini_coefficient, 4),
                'total_samples': len(y),
                'num_classes': len(value_counts)
            },
            'recommendations': self._generate_recommendations(value_counts_pct, rare_classes)
        }

        return analysis

    def _calculate_gini_coefficient(self, value_counts: pd.Series) -> float:
        """محاسبه ضریب جینی برای سنجش عدم تعادل"""
        values = value_counts.values
        n = len(values)
        cumulative_sum = 0

        for i in range(n):
            for j in range(n):
                cumulative_sum += abs(values[i] - values[j])

        if cumulative_sum == 0:
            return 0.0

        return cumulative_sum / (2 * n * np.sum(values))

    def _generate_recommendations(self, value_counts_pct: pd.Series, rare_classes: List[str]) -> List[str]:
        """تولید توصیه‌های هوشمند برای مدیریت عدم تعادل"""
        recommendations = []

        if len(rare_classes) > 0:
            recommendations.append(f"⚠️  {len(rare_classes)} کلاس نادر شناسایی شد: {rare_classes}")
            recommendations.append("🔧 پیشنهاد: ادغام کلاس‌های نادر با کلاس‌های مشابه معنایی")

        imbalance_ratio = value_counts_pct.max() / value_counts_pct.min()
        if imbalance_ratio > 10:
            recommendations.append("⚖️  عدم تعادل شدید - پیشنهاد: ترکیب Oversampling و Undersampling")
        elif imbalance_ratio > 5:
            recommendations.append("⚖️  عدم تعادل متوسط - پیشنهاد: SMOTE با پارامترهای محافظه‌کارانه")
        else:
            recommendations.append("⚖️  عدم تعادل ملایم - پیشنهاد: نمونه‌برداری متعادل")

        # توصیه بر اساس تعداد کلاس‌ها
        if len(value_counts_pct) > 5:
            recommendations.append("🎯 کلاس‌های زیاد - پیشنهاد: استفاده از Class Weights در مدل")

        return recommendations

    def handle_rare_classes(self, y: pd.Series, merge_mapping: Dict[str, str] = None) -> pd.Series:
        """مدیریت خودکار کلاس‌های نادر """
        if merge_mapping is None:
            merge_mapping = self.config.CLASS_MERGE_MAPPING

        # بررسی dtype و تبدیل به string اگر نیاز باشد
        if not pd.api.types.is_string_dtype(y):
            y_processed = y.astype(str)
        else:
            y_processed = y.copy()

        value_counts_pct = y_processed.value_counts(normalize=True) * 100

        # استفاده از vectorization
        rare_classes = value_counts_pct[value_counts_pct < self.config.RARE_CLASS_THRESHOLD * 100].index.tolist()

        # فیلتر کردن فقط کلاس‌هایی که در merge_mapping وجود دارند
        classes_to_merge = [cls for cls in rare_classes if cls in merge_mapping]

        if classes_to_merge:
            # ایجاد mapping dictionary یکجا
            merge_dict = {cls: merge_mapping[cls] for cls in classes_to_merge}

            # اعمال همه جایگزینی‌ها یکجا
            y_processed = y_processed.replace(merge_dict)

            merge_operations = [f"{cls} → {merge_mapping[cls]}" for cls in classes_to_merge]
            print(f"🔧 کلاس‌های نادر ادغام شدند: {merge_operations}")

        return y_processed

    def create_visualizations(self, y_original: pd.Series, y_balanced: Dict[str, pd.Series], output_path: str):
        """ایجاد مصورسازی‌های تحلیلی"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('تحلیل عدم تعادل کلاس‌ها', fontsize=16, fontweight='bold')

            # نمودار ۱: توزیع اصلی
            self._plot_class_distribution(axes[0, 0], y_original, 'توزیع اصلی کلاس‌ها')

            # نمودار ۲: مقایسه توزیع‌ها
            self._plot_comparison(axes[0, 1], y_original, y_balanced)

            # نمودار ۳: معیارهای عدم تعادل
            self._plot_simple_metrics(axes[1, 0], y_original, y_balanced)

            # نمودار ۴: توصیه‌ها
            self._plot_simple_recommendations(axes[1, 1], y_original)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ مصورسازی‌ها ذخیره شد: {output_path}")

        except Exception as e:
            print(f"⚠️  خطا در ایجاد مصورسازی‌ها: {e}")
            # ایجاد یک مصورسازی ساده‌تر به عنوان fallback
            self._create_simple_visualization(y_original, y_balanced, output_path)

    def _plot_class_distribution(self, ax, y: pd.Series, title: str):
        """پلات توزیع کلاس‌ها"""
        counts = y.value_counts()
        percentages = (y.value_counts(normalize=True) * 100).round(2)

        bars = ax.bar(range(len(counts)), counts.values, color='skyblue', alpha=0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45)

        # اضافه کردن اعداد روی نمودار
        for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
            ax.text(i, count + max(counts.values) * 0.01, f'{count}\n({pct}%)',
                    ha='center', va='bottom', fontsize=9)

    def _plot_comparison(self, ax, y_original: pd.Series, y_balanced: Dict[str, pd.Series]):
        """پلات مقایسه توزیع‌ها"""
        strategies = ['اصلی'] + list(y_balanced.keys())
        balanced_data = [y_original] + list(y_balanced.values())

        imbalance_ratios = []
        for y_data in balanced_data:
            counts = y_data.value_counts()
            imbalance_ratios.append(counts.max() / counts.min())

        ax.bar(strategies, imbalance_ratios, color=['red', 'green', 'blue', 'orange'])
        ax.set_title('نسبت عدم تعادل در استراتژی‌های مختلف', fontweight='bold')
        ax.set_ylabel('نسبت عدم تعادل')
        ax.tick_params(axis='x', rotation=45)

        for i, ratio in enumerate(imbalance_ratios):
            ax.text(i, ratio + max(imbalance_ratios) * 0.01, f'{ratio:.1f}',
                    ha='center', va='bottom')

    def _plot_simple_metrics(self, ax, y_original: pd.Series, y_balanced: Dict[str, pd.Series]):
        """پلات ساده معیارهای عدم تعادل"""
        try:
            metrics_data = []
            labels = ['اصلی']

            # معیارهای اصلی
            orig_counts = y_original.value_counts()
            metrics_data.append({
                'تعداد کلاس‌ها': len(orig_counts),
                'نسبت عدم تعادل': orig_counts.max() / orig_counts.min() if orig_counts.min() > 0 else 0,
                'تعداد نمونه‌ها': len(y_original)
            })

            # معیارهای استراتژی‌های متعادل‌شده
            for strategy, y_data in y_balanced.items():
                counts = y_data.value_counts()
                metrics_data.append({
                    'تعداد کلاس‌ها': len(counts),
                    'نسبت عدم تعادل': counts.max() / counts.min() if counts.min() > 0 else 0,
                    'تعداد نمونه‌ها': len(y_data)
                })
                labels.append(strategy)

            # ایجاد نمودار میله‌ای گروهی
            x = np.arange(len(metrics_data))
            width = 0.25

            ax.bar(x - width, [m['تعداد کلاس‌ها'] for m in metrics_data], width, label='تعداد کلاس‌ها')
            ax.bar(x, [m['نسبت عدم تعادل'] for m in metrics_data], width, label='نسبت عدم تعادل')
            ax.bar(x + width, [m['تعداد نمونه‌ها'] for m in metrics_data], width, label='تعداد نمونه‌ها')

            ax.set_title('معیارهای عدم تعادل', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()

        except Exception as e:
            ax.text(0.5, 0.5, f"خطا در رسم معیارها: {e}", ha='center', va='center')
            ax.set_title('معیارهای عدم تعادل')

    def _plot_simple_recommendations(self, ax, y_original: pd.Series):
        """پلات ساده توصیه‌ها"""
        try:
            analysis = self.analyze_class_distribution(y_original)
            recommendations = analysis.get('recommendations', [])

            ax.axis('off')  # غیرفعال کردن محورها
            ax.set_title('توصیه‌های مدیریت عدم تعادل', fontweight='bold')

            # نمایش توصیه‌ها به صورت متن
            text = "\n".join([f"• {rec}" for rec in recommendations])
            if not text.strip():
                text = "• هیچ توصیه‌ای موجود نیست"

            ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', linespacing=1.5)

        except Exception as e:
            ax.text(0.5, 0.5, f"خطا در نمایش توصیه‌ها: {e}", ha='center', va='center')
            ax.set_title('توصیه‌ها')

    def _create_simple_visualization(self, y_original: pd.Series, y_balanced: Dict[str, pd.Series], output_path: str):
        """ایجاد مصورسازی ساده‌تر به عنوان fallback"""
        try:
            plt.figure(figsize=(12, 8))

            # نمودار توزیع اصلی
            plt.subplot(2, 2, 1)
            y_original.value_counts().plot(kind='bar', color='skyblue')
            plt.title('توزیع اصلی کلاس‌ها')
            plt.xticks(rotation=45)

            # نمودار مقایسه‌ای
            plt.subplot(2, 2, 2)
            strategies = ['اصلی'] + list(y_balanced.keys())
            sample_counts = [len(y_original)] + [len(y) for y in y_balanced.values()]
            plt.bar(strategies, sample_counts, color=['red', 'green', 'blue'])
            plt.title('تعداد نمونه‌ها در هر استراتژی')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ مصورسازی ساده ذخیره شد: {output_path}")

        except Exception as e:
            print(f"❌ خطا در ایجاد مصورسازی ساده: {e}")