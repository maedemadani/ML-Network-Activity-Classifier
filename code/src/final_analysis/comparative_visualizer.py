import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib as mpl

# تنظیم فونت برای فارسی
mpl.rcParams['font.family'] = 'DejaVu Sans'


class ComparativeVisualizer:
    """مصورسازی مقایسه‌ای مدل‌ها"""

    def __init__(self, config):
        self.config = config
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    def create_f1_comparison_charts(self, model_summary: pd.DataFrame, output_dir: Path):
        """ایجاد نمودارهای مقایسه‌ای F1"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('مقایسه عملکرد مدل‌ها در شناسایی تهدیدات شبکه',
                     fontsize=16, fontweight='bold', y=0.98)

        # نمودار ۱: F1 کلاس‌های امنیتی
        self._plot_minority_f1_comparison(model_summary, axes[0, 0])

        # نمودار ۲: Recall کلاس‌های امنیتی
        self._plot_minority_recall_comparison(model_summary, axes[0, 1])

        # نمودار ۳: امتیاز امنیتی ترکیبی
        self._plot_security_score_comparison(model_summary, axes[1, 0])

        # نمودار ۴: مقایسه دقت کلی و امنیتی
        self._plot_accuracy_security_tradeoff(model_summary, axes[1, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_model_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ایجاد نمودارهای جداگانه برای هر کلاس اقلیت
        self._create_individual_class_charts(model_summary, output_dir)

    # src/final_analysis/comparative_visualizer.py (قسمت‌های اصلاح شده)

    def _plot_minority_f1_comparison(self, df: pd.DataFrame, ax):
        """نمودار مقایسه F1 کلاس‌های اقلیت"""
        plot_data = []
        for _, row in df.iterrows():
            for cls in self.config.MINORITY_CLASSES:
                plot_data.append({
                    'مدل': f"{row['model']} ({row['dataset']})",
                    'کلاس': f'کلاس {cls}',
                    'F1-Score': row[f'f1_class_{cls}']
                })

        plot_df = pd.DataFrame(plot_data)

        if not plot_df.empty:
            # محدود کردن palette به تعداد کلاس‌ها
            palette = self.colors[:len(self.config.MINORITY_CLASSES)]
            sns.barplot(data=plot_df, x='مدل', y='F1-Score', hue='کلاس', ax=ax, palette=palette)
            ax.set_title('F1-Score کلاس‌های امنیتی', fontweight='bold')

            # تنظیم برچسب‌ها با FixedLocator برای رفع warning
            ticks = list(range(len(plot_df['مدل'].unique())))
            ax.set_xticks(ticks)
            ax.set_xticklabels(plot_df['مدل'].unique(), rotation=45, ha='right')
            ax.legend(title='کلاس‌های امنیتی')

    def _plot_minority_recall_comparison(self, df: pd.DataFrame, ax):
        """نمودار مقایسه Recall کلاس‌های اقلیت"""
        plot_data = []
        for _, row in df.iterrows():
            for cls in self.config.MINORITY_CLASSES:
                plot_data.append({
                    'مدل': f"{row['model']} ({row['dataset']})",
                    'کلاس': f'کلاس {cls}',
                    'Recall': row[f'recall_class_{cls}']
                })

        plot_df = pd.DataFrame(plot_data)

        if not plot_df.empty:
            # محدود کردن palette به تعداد کلاس‌ها
            palette = self.colors[1:1 + len(self.config.MINORITY_CLASSES)]
            sns.barplot(data=plot_df, x='مدل', y='Recall', hue='کلاس', ax=ax, palette=palette)
            ax.set_title('Recall کلاس‌های امنیتی', fontweight='bold')

            # تنظیم برچسب‌ها با FixedLocator برای رفع warning
            ticks = list(range(len(plot_df['مدل'].unique())))
            ax.set_xticks(ticks)
            ax.set_xticklabels(plot_df['مدل'].unique(), rotation=45, ha='right')
            ax.legend(title='کلاس‌های امنیتی')

    def _plot_security_score_comparison(self, df: pd.DataFrame, ax):
        """نمودار مقایسه امتیاز امنیتی"""
        if not df.empty:
            # محدود کردن palette به تعداد datasetها
            unique_datasets = df['dataset'].unique()
            palette = self.colors[:len(unique_datasets)]

            sns.barplot(data=df, x='model', y='security_score', hue='dataset', ax=ax, palette=palette)
            ax.set_title('امتیاز امنیتی ترکیبی مدل‌ها', fontweight='bold')
            ax.set_xlabel('مدل')
            ax.set_ylabel('امتیاز امنیتی')
            ax.legend(title='داده آموزشی')

    def _plot_accuracy_security_tradeoff(self, df: pd.DataFrame, ax):
        """نمودار Trade-off بین دقت و امنیت"""
        scatter = ax.scatter(df['accuracy'], df['security_score'],
                             c=df['f1_minority_mean'], cmap='viridis',
                             s=100, alpha=0.7)

        # اضافه کردن برچسب‌ها
        for i, row in df.iterrows():
            ax.annotate(f"{row['model']}({row['dataset'][:3]})",
                        (row['accuracy'], row['security_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('دقت کلی (Accuracy)')
        ax.set_ylabel('امتیاز امنیتی')
        ax.set_title('تعادل بین دقت کلی و امنیت', fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='میانگین F1 کلاس‌های اقلیت')

    def _create_individual_class_charts(self, df: pd.DataFrame, output_dir: Path):
        """ایجاد نمودارهای جداگانه برای هر کلاس"""
        for cls in self.config.MINORITY_CLASSES:
            fig, ax = plt.subplots(figsize=(12, 6))

            # گروه‌بندی داده‌ها
            plot_data = []
            for _, row in df.iterrows():
                plot_data.append({
                    'مدل': row['model'],
                    'داده': row['dataset'],
                    'F1-Score': row[f'f1_class_{cls}'],
                    'Recall': row[f'recall_class_{cls}']
                })

            plot_df = pd.DataFrame(plot_data)

            # نمودار گروهی
            x = np.arange(len(plot_df['مدل'].unique()))
            width = 0.35

            for i, dataset in enumerate(plot_df['داده'].unique()):
                dataset_data = plot_df[plot_df['داده'] == dataset]
                f1_scores = dataset_data['F1-Score'].values
                recalls = dataset_data['Recall'].values

                ax.bar(x + i * width, f1_scores, width, label=f'F1 ({dataset})', alpha=0.7)
                ax.bar(x + i * width, recalls, width, label=f'Recall ({dataset})', alpha=0.7, bottom=f1_scores)

            ax.set_xlabel('مدل')
            ax.set_ylabel('مقدار')
            ax.set_title(f'عملکرد مدل‌ها برای کلاس {cls}', fontweight='bold')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(plot_df['مدل'].unique())
            ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / f'class_{cls}_performance.png', dpi=300, bbox_inches='tight')
            plt.close()