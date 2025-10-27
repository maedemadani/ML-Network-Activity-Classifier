import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any


class ModelSelector:
    """انتخاب‌کننده مدل نهایی"""

    def __init__(self, config):
        self.config = config
        self.selection_results = {}

    def select_best_model(self, model_summary: pd.DataFrame) -> Dict[str, Any]:
        """انتخاب بهترین مدل بر اساس معیارهای تعریف شده"""
        print("🏆 در حال انتخاب بهترین مدل...")

        # فیلتر مدل‌ها بر اساس معیارهای minimum
        filtered_models = self._filter_models_by_criteria(model_summary)

        if filtered_models.empty:
            print("⚠️  هیچ مدلی معیارهای minimum را نداشت. در حال استفاده از همه مدل‌ها...")
            filtered_models = model_summary

        # رتبه‌بندی مدل‌ها
        ranked_models = self._rank_models(filtered_models)

        # انتخاب مدل برتر و مدل‌های پشتیبان
        best_model = ranked_models.iloc[0]
        runner_ups = ranked_models.iloc[1:3] if len(ranked_models) > 1 else pd.DataFrame()

        selection_info = {
            'selected_model': self._format_model_info(best_model),
            'runner_ups': [self._format_model_info(model) for _, model in runner_ups.iterrows()],
            'selection_criteria_used': self.config.SELECTION_CRITERIA,
            'ranking_metrics': ['security_score', 'f1_minority_mean', 'recall_minority_mean']
        }

        # تحلیل trade-off
        selection_info['trade_off_analysis'] = self._analyze_trade_offs(best_model, model_summary)

        self.selection_results = selection_info
        return selection_info

    def _filter_models_by_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """فیلتر مدل‌ها بر اساس معیارهای minimum"""
        criteria = self.config.SELECTION_CRITERIA

        filtered = df.copy()

        if 'f1_minority_mean' in criteria:
            filtered = filtered[filtered['f1_minority_mean'] >= criteria['f1_minority_mean']]

        if 'min_recall_minority' in criteria:
            filtered = filtered[filtered['recall_minority_mean'] >= criteria['min_recall_minority']]

        if 'max_inference_time_ms' in criteria:
            filtered = filtered[filtered['inference_time_ms'] <= criteria['max_inference_time_ms']]

        if 'max_model_size_mb' in criteria:
            filtered = filtered[filtered['model_size_mb'] <= criteria['max_model_size_mb']]

        return filtered

    def _rank_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """رتبه‌بندی مدل‌ها"""
        # محاسبه امتیاز نهایی
        df['final_score'] = (
                df['security_score'] * 0.4 +
                df['f1_minority_mean'] * 0.3 +
                df['recall_minority_mean'] * 0.2 +
                df['threat_detection_rate'] * 0.1
        )

        # رتبه‌بندی نزولی
        ranked = df.sort_values('final_score', ascending=False)
        return ranked

    def _format_model_info(self, model_row: pd.Series) -> Dict[str, Any]:
        """فرمت‌دهی اطلاعات مدل"""
        return {
            'model': model_row['model'],
            'dataset': model_row['dataset'],
            'metrics': {
                'accuracy': float(model_row['accuracy']),
                'f1_macro': float(model_row['f1_macro']),
                'f1_minority_mean': float(model_row['f1_minority_mean']),
                'recall_minority_mean': float(model_row['recall_minority_mean']),
                'security_score': float(model_row['security_score']),
                'threat_detection_rate': float(model_row['threat_detection_rate'])
            },
            'minority_class_performance': {
                f'class_{cls}': {
                    'f1': float(model_row[f'f1_class_{cls}']),
                    'recall': float(model_row[f'recall_class_{cls}']),
                    'precision': float(model_row[f'precision_class_{cls}'])
                } for cls in self.config.MINORITY_CLASSES
            }
        }

    def _analyze_trade_offs(self, selected_model: pd.Series, all_models: pd.DataFrame) -> Dict[str, Any]:
        """تحلیل trade-off‌های مدل انتخاب‌شده"""
        best_accuracy = all_models['accuracy'].max()
        best_f1_macro = all_models['f1_macro'].max()

        trade_offs = {
            'accuracy_tradeoff': float(best_accuracy - selected_model['accuracy']),
            'f1_macro_tradeoff': float(best_f1_macro - selected_model['f1_macro']),
            'security_improvement': float(selected_model['security_score'] - all_models['security_score'].mean()),
            'explanation': self._generate_tradeoff_explanation(selected_model, best_accuracy, best_f1_macro)
        }

        return trade_offs

    def _generate_tradeoff_explanation(self, selected_model: pd.Series,
                                       best_accuracy: float, best_f1_macro: float) -> str:
        """تولید توضیح trade-off"""
        accuracy_diff = best_accuracy - selected_model['accuracy']
        f1_diff = best_f1_macro - selected_model['f1_macro']

        explanations = []

        if accuracy_diff > 0.05:
            explanations.append(f"دقت کلی {accuracy_diff:.3f} کمتر از بهترین مدل است")
        elif accuracy_diff > 0.02:
            explanations.append(f"دقت کلی کمی ({accuracy_diff:.3f}) کمتر است")
        else:
            explanations.append("دقت کلی تقریباً بهینه است")

        if f1_diff > 0.05:
            explanations.append(f"F1 کلی {f1_diff:.3f} کمتر از بهترین مدل است")
        elif f1_diff > 0.02:
            explanations.append(f"F1 کلی کمی ({f1_diff:.3f}) کمتر است")
        else:
            explanations.append("F1 کلی تقریباً بهینه است")

        security_improvement = selected_model['security_score'] - selected_model['f1_macro']
        if security_improvement > 0.1:
            explanations.append("امتیاز امنیتی به طور قابل توجهی بهبود یافته است")
        elif security_improvement > 0.05:
            explanations.append("امتیاز امنیتی moderately بهبود یافته است")
        else:
            explanations.append("تعادل خوبی بین امنیت و دقت کلی برقرار است")

        return ". ".join(explanations)

    def save_selection_results(self, output_dir: Path):
        """ذخیره نتایج انتخاب"""
        json_path = output_dir / "selected_model.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.selection_results, f, indent=2, ensure_ascii=False)

        # ایجاد گزارش متنی
        self._create_selection_report(output_dir)

        print(f"🏆 نتایج انتخاب مدل در {output_dir} ذخیره شدند")

    def _create_selection_report(self, output_dir: Path):
        """ایجاد گزارش متنی انتخاب مدل"""
        report_path = output_dir / "model_selection_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("گزارش انتخاب مدل نهایی\n")
            f.write("=" * 60 + "\n\n")

            selected = self.selection_results['selected_model']
            f.write(f"مدل انتخاب‌شده: {selected['model']} (داده: {selected['dataset']})\n\n")

            f.write("معیارهای عملکرد:\n")
            for metric, value in selected['metrics'].items():
                f.write(f"  {metric}: {value:.3f}\n")

            f.write("\nعملکرد کلاس‌های امنیتی:\n")
            for cls, perf in selected['minority_class_performance'].items():
                f.write(f"  {cls}:\n")
                f.write(f"    F1: {perf['f1']:.3f}\n")
                f.write(f"    Recall: {perf['recall']:.3f}\n")
                f.write(f"    Precision: {perf['precision']:.3f}\n")

            f.write("\nتحلیل Trade-off:\n")
            trade_offs = self.selection_results['trade_off_analysis']
            f.write(f"  کاهش دقت: {trade_offs['accuracy_tradeoff']:.3f}\n")
            f.write(f"  کاهش F1 کلی: {trade_offs['f1_macro_tradeoff']:.3f}\n")
            f.write(f"  بهبود امنیت: {trade_offs['security_improvement']:.3f}\n")
            f.write(f"  توضیح: {trade_offs['explanation']}\n")

            if self.selection_results['runner_ups']:
                f.write("\nمدل‌های پشتیبان:\n")
                for i, runner_up in enumerate(self.selection_results['runner_ups'], 1):
                    f.write(
                        f"  {i}. {runner_up['model']} ({runner_up['dataset']}) - امتیاز امنیتی: {runner_up['metrics']['security_score']:.3f}\n")