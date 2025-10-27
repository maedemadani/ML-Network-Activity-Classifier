import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.utils import resample


class StatisticalAnalyzer:
    """تحلیل آماری نتایج مدل‌ها"""

    def __init__(self, config):
        self.config = config
        self.statistical_results = {}

    def _convert_to_serializable(self, obj: Any) -> Any:
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
        elif isinstance(obj, bool):
            return bool(obj)  # اطمینان از تبدیل به bool استاندارد
        else:
            return obj

    def perform_bootstrap_analysis(self, model_summary: pd.DataFrame) -> Dict[str, Any]:
        """انجام تحلیل بوت‌استرپ برای فاصله اطمینان"""
        bootstrap_results = {}

        for metric in ['f1_minority_mean', 'security_score', 'mean_security_recall',
                       'security_f1', 'accuracy']:
            values = model_summary[metric].values
            bootstrap_ci = self._bootstrap_confidence_interval(values)
            bootstrap_results[metric] = bootstrap_ci

        self.statistical_results['bootstrap'] = bootstrap_results
        return bootstrap_results

    def _bootstrap_confidence_interval(self, values: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """محاسبه فاصله اطمینان بوت‌استرپ"""
        # اگر فقط یک مقدار داریم، فاصله اطمینان معنی ندارد
        if len(values) <= 1:
            return {
                'mean': float(values[0]) if len(values) == 1 else 0.0,
                'std': 0.0,
                'ci_lower': float(values[0]) if len(values) == 1 else 0.0,
                'ci_upper': float(values[0]) if len(values) == 1 else 0.0,
                'confidence_level': confidence,
                'warning': 'Insufficient data for confidence interval'
            }

        bootstrap_stats = []

        for _ in range(self.config.N_BOOTSTRAP_SAMPLES):
            sample = resample(values)
            bootstrap_stats.append(np.mean(sample))

        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_stats, alpha * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'confidence_level': confidence
        }

    def perform_pairwise_tests(self, model_summary: pd.DataFrame) -> Dict[str, Any]:
        """انجام آزمون‌های زوجی بین استراتژی‌ها"""
        pairwise_results = {}

        models = model_summary['model'].unique()
        datasets = model_summary['dataset'].unique()

        for model in models:
            model_data = model_summary[model_summary['model'] == model]
            if len(model_data) < 2:
                continue

            pairwise_results[model] = {}

            # مقایسه هر جفت استراتژی
            for i, ds1 in enumerate(datasets):
                for j, ds2 in enumerate(datasets):
                    if i >= j:
                        continue

                    ds1_data = model_data[model_data['dataset'] == ds1]
                    ds2_data = model_data[model_data['dataset'] == ds2]

                    if len(ds1_data) > 0 and len(ds2_data) > 0:
                        comparison = self._compare_strategies(
                            ds1_data.iloc[0], ds2_data.iloc[0], ds1, ds2
                        )
                        pairwise_results[model][f'{ds1}_vs_{ds2}'] = comparison

        self.statistical_results['pairwise'] = pairwise_results
        return pairwise_results

    def _compare_strategies(self, strategy1: pd.Series, strategy2: pd.Series,
                            name1: str, name2: str) -> Dict[str, Any]:
        """مقایسه دو استراتژی"""
        comparison = {
            'strategy1': name1,
            'strategy2': name2,
            'compared_metrics': []
        }

        metrics_to_compare = [
            'f1_minority_mean',
            'security_score',
            'mean_security_recall',
            'security_f1',
            'accuracy'
        ]

        for metric in metrics_to_compare:
            val1 = strategy1[metric]
            val2 = strategy2[metric]

            # مدیریت مقادیر یکسان (برای جلوگیری از تقسیم بر صفر)
            if val1 == val2:
                comparison['compared_metrics'].append({
                    'metric': metric,
                    f'{name1}_value': float(val1),
                    f'{name2}_value': float(val2),
                    'difference': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'effect_size': 0.0,
                    'note': 'identical_values'
                })
                continue

            # آزمون t زوجی (ساده‌شده)
            try:
                t_stat, p_value = stats.ttest_rel([val1], [val2])
                effect_size = abs(val1 - val2) / max(np.std([val1, val2]), 0.001)  # جلوگیری از تقسیم بر صفر

                comparison['compared_metrics'].append({
                    'metric': metric,
                    f'{name1}_value': float(val1),
                    f'{name2}_value': float(val2),
                    'difference': float(val1 - val2),
                    'p_value': float(p_value),
                    'significant': bool(p_value < self.config.STATISTICAL_ALPHA),
                    'effect_size': float(effect_size)
                })
            except Exception as e:
                comparison['compared_metrics'].append({
                    'metric': metric,
                    f'{name1}_value': float(val1),
                    f'{name2}_value': float(val2),
                    'difference': float(val1 - val2),
                    'p_value': None,
                    'significant': False,
                    'effect_size': float(abs(val1 - val2)),
                    'error': str(e)
                })

        return comparison

    def save_statistical_results(self, output_dir: Path):
        """ذخیره نتایج آماری"""
        # تبدیل به انواع قابل سریالایز
        serializable_results = self._convert_to_serializable(self.statistical_results)

        # ذخیره JSON
        json_path = output_dir / "statistical_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # ایجاد گزارش متنی
        self._create_statistical_report(output_dir)

        print(f"📈 نتایج آماری در {output_dir} ذخیره شدند")

    def _create_statistical_report(self, output_dir: Path):
        """ایجاد گزارش متنی تحلیل آماری"""
        report_path = output_dir / "statistical_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("گزارش تحلیل آماری نتایج مدل‌ها\n")
            f.write("=" * 60 + "\n\n")

            # نتایج بوت‌استرپ
            f.write("فاصله اطمینان بوت‌استرپ (۹۵٪):\n")
            if 'bootstrap' in self.statistical_results:
                for metric, ci in self.statistical_results['bootstrap'].items():
                    f.write(f"  {metric}:\n")
                    f.write(f"    میانگین: {ci['mean']:.3f}\n")
                    f.write(f"    انحراف معیار: {ci['std']:.3f}\n")
                    if 'warning' in ci:
                        f.write(f"    ⚠️  {ci['warning']}\n")
                    else:
                        f.write(f"    فاصله اطمینان: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]\n\n")

            # نتایج آزمون‌های زوجی
            f.write("آزمون‌های زوجی بین استراتژی‌ها:\n")
            if 'pairwise' in self.statistical_results:
                for model, comparisons in self.statistical_results['pairwise'].items():
                    f.write(f"\n  مدل {model}:\n")
                    for comp_name, comp_data in comparisons.items():
                        f.write(f"    مقایسه {comp_name}:\n")
                        for metric_comp in comp_data['compared_metrics']:
                            if 'error' in metric_comp:
                                f.write(f"      {metric_comp['metric']}: خطا - {metric_comp['error']}\n")
                            elif 'note' in metric_comp and metric_comp['note'] == 'identical_values':
                                f.write(f"      {metric_comp['metric']}: مقادیر یکسان\n")
                            else:
                                significance = "✅ معنی‌دار" if metric_comp['significant'] else "❌ غیر معنی‌دار"
                                f.write(
                                    f"      {metric_comp['metric']}: p-value={metric_comp['p_value']:.4f} ({significance})\n")