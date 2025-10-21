import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any ,List
from pathlib import Path


class BalanceReporter:
    """گزارش‌ده هوشمند مدیریت عدم تعادل"""

    def __init__(self, config):
        self.config = config



    def _generate_next_steps(self) -> List[str]:
        """تولید مراحل بعدی برای فاز ۴"""
        return [
            "🎯 فاز ۴: آموزش مدل‌ها روی هر سه استراتژی (Original, Undersampled, Oversampled)",
            "📊 ارزیابی مدل‌ها بر اساس Recall کلاس‌های اقلیت (drop, deny)",
            "🔍 مقایسه عملکرد مدل‌ها روی Test set دست‌نخورده",
            "🏆 انتخاب بهترین استراتژی بر اساس معیارهای امنیتی",
            "💾 ذخیره مدل نهایی برای deployment"
        ]

    def _generate_final_recommendations(self, analysis_report: Dict[str, Any],
                                        sampling_report: Dict[str, Any],
                                        validation_report: Dict[str, Any]) -> List[str]:
        """تولید توصیه‌های نهایی"""
        recommendations = []

        # تحلیل توزیع کلاس‌ها
        imbalance_ratio = analysis_report.get('imbalance_metrics', {}).get('imbalance_ratio', 1)

        if imbalance_ratio > 50:
            recommendations.append("🎯 عدم تعادل شدید - پیشنهاد: استفاده از ترکیب Ensemble + Class Weights")
        elif imbalance_ratio > 10:
            recommendations.append("🎯 عدم تعابل متوسط - پیشنهاد: SMOTE + RandomUnderSampler ترکیبی")
        else:
            recommendations.append("🎯 عدم تعادل ملایم - پیشنهاد: نمونه‌برداری ساده کافی است")

        # تحلیل استراتژی‌های نمونه‌برداری
        if 'oversampling' in sampling_report:
            method = sampling_report['oversampling'].get('method', '')
            if 'RandomOverSampler' in method:
                recommendations.append("⚠️  برای کلاس‌های با نمونه کم از RandomOverSampler استفاده شد - کیفیت پایین‌تر")
            else:
                recommendations.append("✅ از SMOTE برای تولید نمونه‌های مصنوعی استفاده شد")

        # تحلیل اعتبارسنجی
        for strategy, validation in validation_report.items():
            basic_checks = validation.get('basic_checks', {})
            if basic_checks.get('all_passed', False):
                recommendations.append(f"✅ استراتژی {strategy} تمام بررسی‌های پایه را گذراند")
            else:
                recommendations.append(f"⚠️  استراتژی {strategy} مشکلاتی در بررسی‌های پایه دارد")

        recommendations.append("🔮 برای فاز ۴: تست تمام استراتژی‌ها و انتخاب بهترین بر اساس Recall کلاس‌های اقلیت")

        return recommendations

    def _create_summary_report(self, comprehensive_report: Dict[str, Any], output_dir: Path):
        """ایجاد گزارش خلاصه متنی"""
        summary_path = output_dir / "balance_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("خلاصه مدیریت عدم تعادل کلاس‌ها\n")
            f.write("=" * 60 + "\n\n")

            # اطلاعات کلی
            analysis = comprehensive_report['class_analysis']
            f.write("📊 تحلیل توزیع کلاس‌ها:\n")
            for cls, pct in analysis.get('class_percentages', {}).items():
                f.write(f"   • {cls}: {pct}%\n")

            f.write(f"\n⚖️  نسبت عدم تعادل: {analysis.get('imbalance_metrics', {}).get('imbalance_ratio', 1):.1f}\n")

            # استراتژی‌ها
            f.write("\n🔧 استراتژی‌های اعمال شده:\n")
            sampling = comprehensive_report['sampling_strategies']
            for strategy in sampling.keys():
                f.write(f"   • {strategy}\n")

            # توصیه‌ها
            f.write("\n🎯 توصیه‌های اصلی:\n")
            for rec in comprehensive_report['recommendations']:
                f.write(f"   • {rec}\n")

    def _convert_to_serializable(self, obj: Any) -> Any:
        """تبدیل به انواع قابل سریالایز"""
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
        elif pd.isna(obj):
            return None
        else:
            return obj

    def _calculate_quality_score(self, validation: Dict[str, Any]) -> float:
        """محاسبه نمره کیفیت برای یک استراتژی"""
        try:
            score = 0.0
            total_checks = 0

            # بررسی معیارهای پایه
            basic_checks = validation.get('basic_checks', {})
            if not basic_checks.get('has_nulls', True):
                score += 1
            total_checks += 1

            if basic_checks.get('shape_consistent', False):
                score += 1
            total_checks += 1

            if basic_checks.get('data_types_consistent', False):
                score += 1
            total_checks += 1

            if basic_checks.get('class_diversity', False):
                score += 1
            total_checks += 1

            # بررسی توزیع
            distribution_checks = validation.get('distribution_checks', {})
            if distribution_checks.get('class_preservation', False):
                score += 1
            total_checks += 1

            if distribution_checks.get('min_samples_per_class', False):
                score += 1
            total_checks += 1

            # بررسی نشت داده
            leakage_checks = validation.get('leakage_checks', {})
            if leakage_checks.get('no_data_leakage', False):
                score += 1
            total_checks += 1

            quality_score = score / total_checks if total_checks > 0 else 0.0

            print(f"   🎯 کیفیت: {quality_score:.2f} (has_nulls: {basic_checks.get('has_nulls')})")

            return quality_score

        except Exception as e:
            print(f"⚠️  خطا در محاسبه کیفیت: {e}")
            return 0.5  # مقدار پیش‌فرض

    def _calculate_performance_metrics(self, analysis_report: Dict[str, Any],
                                       validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه معیارهای عملکرد"""
        original_imbalance = analysis_report.get('imbalance_metrics', {}).get('imbalance_ratio', 1)

        metrics = {}
        for strategy, validation in validation_report.items():
            if strategy != 'original':
                balanced_imbalance = validation.get('distribution_checks', {}).get('balanced_imbalance_ratio', 1)
                improvement = original_imbalance / balanced_imbalance if balanced_imbalance > 0 else 1

                metrics[strategy] = {
                    'imbalance_improvement': round(improvement, 2),
                    'quality_score': round(self._calculate_quality_score(validation), 2),
                    'recommended_for': self._get_recommended_use_case(strategy, improvement)
                }

        return metrics

    def _get_recommended_use_case(self, strategy: str, improvement: float) -> str:
        """تعیین مورد استفاده توصیه شده برای استراتژی"""
        if strategy == 'undersampled':
            return "داده‌های با حجم زیاد و زمان آموزش کوتاه"
        elif strategy == 'oversampled':
            if improvement > 10:
                return "داده‌های با عدم تعادل شدید و اهمیت کلاس‌های اقلیت"
            else:
                return "داده‌های با عدم تعادل متوسط"
        else:
            return "موارد عمومی"

    def _create_visual_summary(self, comprehensive_report: Dict[str, Any], output_dir: Path):
        """ایجاد خلاصه مصور"""
        try:
            # ایجاد یک نمودار ساده برای نمایش بهبود عدم تعادل
            strategies = []
            imbalance_ratios = []

            for strategy, metrics in comprehensive_report.get('performance_metrics', {}).items():
                strategies.append(strategy)
                imbalance_ratios.append(metrics.get('imbalance_improvement', 1))

            if strategies:
                plt.figure(figsize=(10, 6))
                plt.bar(strategies, imbalance_ratios, color=['green', 'blue', 'orange'])
                plt.title('Improvement in Imbalance Ratio by Strategy')
                plt.ylabel('Improvement Ratio')
                plt.xlabel('Sampling Strategy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'imbalance_improvement.png', dpi=300, bbox_inches='tight')
                plt.close()

                print(f"   ✅ نمودار بهبود عدم تعادل ذخیره شد")
        except Exception as e:
            print(f"⚠️  خطا در ایجاد نمودار: {e}")

    def generate_comprehensive_report(self,
                                      analysis_report: Dict[str, Any],
                                      sampling_report: Dict[str, Any],
                                      validation_report: Dict[str, Any],
                                      output_dir: Path) -> Dict[str, Any]:
        """تولید گزارش جامع"""

        # محاسبه آمارهای اضافی
        total_samples = analysis_report.get('imbalance_metrics', {}).get('total_samples', 0)
        num_classes = analysis_report.get('imbalance_metrics', {}).get('num_classes', 0)

        comprehensive_report = {
            'metadata': {
                'phase': 'Class Balancing - Phase 3',
                'timestamp': pd.Timestamp.now().isoformat(),
                'config': {
                    'test_size': self.config.TEST_SIZE,
                    'rare_class_threshold': self.config.RARE_CLASS_THRESHOLD,
                    'random_state': self.config.RANDOM_STATE
                },
                'statistics': {
                    'total_samples': total_samples,
                    'num_classes': num_classes,
                    'test_samples': int(total_samples * self.config.TEST_SIZE),
                    'train_samples': int(total_samples * (1 - self.config.TEST_SIZE))
                }
            },
            'class_analysis': analysis_report,
            'sampling_strategies': sampling_report,
            'validation_results': validation_report,
            'performance_metrics': self._calculate_performance_metrics(analysis_report, validation_report),
            'recommendations': self._generate_final_recommendations(analysis_report, sampling_report,
                                                                    validation_report),
            'next_steps': self._generate_next_steps()
        }

        # ذخیره گزارش
        report_path = output_dir / "class_balance_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_to_serializable(comprehensive_report), f, indent=2, ensure_ascii=False)

        # ایجاد گزارش خلاصه و مصورسازی
        self._create_summary_report(comprehensive_report, output_dir)
        self._create_visual_summary(comprehensive_report, output_dir)

        return comprehensive_report