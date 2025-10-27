import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelEvaluator:
    """ارزیابی‌کننده جامع مدل‌های طبقه‌بندی"""

    def __init__(self, config):
        self.config = config
        self.evaluation_results = {}
        self.security_metrics = {}

    def load_models_and_preprocessors(self, strategy_name: str) -> Tuple[Dict[str, Any], Any]:
        """بارگذاری مدل‌ها و preprocessor یک استراتژی"""
        base_path = Path(self.config.MODEL_SAVE_PATHS['base_models']) / strategy_name
        preprocessor_path = Path(self.config.MODEL_SAVE_PATHS['preprocessors']) / f"{strategy_name}_preprocessor.pkl"

        models = {}
        for model_file in base_path.glob("*.pkl"):
            model_name = model_file.stem
            models[model_name] = joblib.load(model_file)

        preprocessor = joblib.load(preprocessor_path) if preprocessor_path.exists() else None

        return models, preprocessor

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                        strategy_name: str) -> Dict[str, Any]:
        """ارزیابی تمام مدل‌های یک استراتژی روی تست set"""
        print(f"📊 ارزیابی مدل‌های استراتژی: {strategy_name}")

        # بارگذاری مدل‌ها و preprocessor
        models, preprocessor = self.load_models_and_preprocessors(strategy_name)

        if preprocessor is None:
            print(f"   ⚠️  Preprocessor برای {strategy_name} یافت نشد")
            return {}

        # پیش‌پردازش داده تست
        X_test_processed = preprocessor.transform(X_test)

        evaluation_results = {}

        for model_name, model in models.items():
            print(f"   🔍 ارزیابی {model_name}...")

            try:
                # پیش‌بینی
                y_pred = model.predict(X_test_processed)
                y_pred_proba = model.predict_proba(X_test_processed) if hasattr(model, 'predict_proba') else None

                # محاسبه معیارها
                metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)

                # محاسبه معیارهای امنیتی
                security_metrics = self._calculate_security_metrics(y_test, y_pred)

                evaluation_results[model_name] = {
                    'general_metrics': metrics,
                    'security_metrics': security_metrics,
                    'predictions': {
                        'y_true': y_test.values.tolist(),
                        'y_pred': y_pred.tolist(),
                        'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
                    }
                }

                print(f"   ✅ {model_name} ارزیابی شد (Accuracy: {metrics['accuracy']:.3f})")

            except Exception as e:
                evaluation_results[model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                print(f"   ❌ خطا در ارزیابی {model_name}: {e}")

        self.evaluation_results[strategy_name] = evaluation_results
        return evaluation_results

    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """محاسبه معیارهای جامع ارزیابی"""
        metrics = {}

        # معیارهای کلی
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # معیارهای هر کلاس
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            cls_idx = y_true == cls
            if np.sum(cls_idx) > 0:
                metrics[f'precision_class_{cls}'] = precision_score(y_true, y_pred, labels=[cls], average=None)[0]
                metrics[f'recall_class_{cls}'] = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
                metrics[f'f1_class_{cls}'] = f1_score(y_true, y_pred, labels=[cls], average=None)[0]

        return metrics

    def _calculate_security_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """محاسبه معیارهای امنیتی ویژه"""
        # فرض: کلاس‌های ۱ و ۲ کلاس‌های امنیتی هستند (deny/drop)
        security_classes = [1, 2]

        security_metrics = {}

        for cls in security_classes:
            cls_mask = y_true == cls
            if np.sum(cls_mask) > 0:
                # Recall برای کلاس‌های امنیتی (مهم‌ترین معیار)
                security_metrics[f'recall_class_{cls}'] = recall_score(
                    y_true, y_pred, labels=[cls], average=None
                )[0]

                # Precision برای کلاس‌های امنیتی
                security_metrics[f'precision_class_{cls}'] = precision_score(
                    y_true, y_pred, labels=[cls], average=None
                )[0]

        # میانگین معیارهای امنیتی
        security_recalls = [security_metrics.get(f'recall_class_{cls}', 0) for cls in security_classes]
        security_precisions = [security_metrics.get(f'precision_class_{cls}', 0) for cls in security_classes]

        security_metrics['mean_security_recall'] = np.mean(security_recalls)
        security_metrics['mean_security_precision'] = np.mean(security_precisions)
        security_metrics['security_f1'] = 2 * (
                    security_metrics['mean_security_recall'] * security_metrics['mean_security_precision']) / \
                                          (security_metrics['mean_security_recall'] + security_metrics[
                                              'mean_security_precision'] + 1e-8)

        # نرخ شناسایی تهدید
        threat_mask = np.isin(y_true, security_classes)
        if np.sum(threat_mask) > 0:
            threat_predictions = y_pred[threat_mask]
            correct_threat_predictions = np.sum(np.isin(threat_predictions, security_classes))
            security_metrics['threat_detection_rate'] = correct_threat_predictions / np.sum(threat_mask)
        else:
            security_metrics['threat_detection_rate'] = 0.0

        return security_metrics

    def create_comparative_analysis(self) -> Dict[str, Any]:
        """ایجاد تحلیل مقایسه‌ای بین تمام مدل‌ها و استراتژی‌ها"""
        comparative_results = {}

        for strategy_name, models_eval in self.evaluation_results.items():
            comparative_results[strategy_name] = {}

            for model_name, eval_result in models_eval.items():
                if 'general_metrics' in eval_result:
                    comparative_results[strategy_name][model_name] = {
                        'accuracy': eval_result['general_metrics']['accuracy'],
                        'f1_macro': eval_result['general_metrics']['f1_macro'],
                        'mean_security_recall': eval_result['security_metrics']['mean_security_recall'],
                        'security_f1': eval_result['security_metrics']['security_f1'],
                        'threat_detection_rate': eval_result['security_metrics']['threat_detection_rate']
                    }

        return comparative_results

    def generate_reports(self, output_dir: Path):
        """تولید گزارش‌های جامع ارزیابی"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # ۱. گزارش مقایسه‌ای
        comparative_analysis = self.create_comparative_analysis()
        comp_path = output_dir / "comparative_analysis.json"
        with open(comp_path, 'w', encoding='utf-8') as f:
            json.dump(comparative_analysis, f, indent=2, ensure_ascii=False)

        # ۲. گزارش تفصیلی هر استراتژی
        for strategy_name, models_eval in self.evaluation_results.items():
            strategy_path = output_dir / f"detailed_report_{strategy_name}.json"
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(models_eval, f, indent=2, ensure_ascii=False)

        # ۳. ایجاد نمودارهای مقایسه‌ای
        self._create_comparative_charts(comparative_analysis, output_dir)

        print(f"📋 گزارش‌های ارزیابی در {output_dir} ذخیره شدند")

    def _create_comparative_charts(self, comparative_analysis: Dict[str, Any], output_dir: Path):
        """ایجاد نمودارهای مقایسه‌ای"""
        strategies = list(comparative_analysis.keys())
        model_names = set()

        for strategy_models in comparative_analysis.values():
            model_names.update(strategy_models.keys())

        model_names = list(model_names)

        # نمودار ۱: مقایسه accuracy
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('مقایسه عملکرد مدل‌ها در استراتژی‌های مختلف', fontsize=16, fontweight='bold')

        # Accuracy
        accuracy_data = []
        for strategy in strategies:
            for model in model_names:
                if model in comparative_analysis[strategy]:
                    accuracy_data.append({
                        'strategy': strategy,
                        'model': model,
                        'accuracy': comparative_analysis[strategy][model]['accuracy']
                    })

        accuracy_df = pd.DataFrame(accuracy_data)
        if not accuracy_df.empty:
            sns.barplot(data=accuracy_df, x='model', y='accuracy', hue='strategy', ax=axes[0, 0])
            axes[0, 0].set_title('دقت (Accuracy) مدل‌ها')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # F1 Macro
        f1_data = []
        for strategy in strategies:
            for model in model_names:
                if model in comparative_analysis[strategy]:
                    f1_data.append({
                        'strategy': strategy,
                        'model': model,
                        'f1_macro': comparative_analysis[strategy][model]['f1_macro']
                    })

        f1_df = pd.DataFrame(f1_data)
        if not f1_df.empty:
            sns.barplot(data=f1_df, x='model', y='f1_macro', hue='strategy', ax=axes[0, 1])
            axes[0, 1].set_title('F1-Score Macro مدل‌ها')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Security Recall
        security_data = []
        for strategy in strategies:
            for model in model_names:
                if model in comparative_analysis[strategy]:
                    security_data.append({
                        'strategy': strategy,
                        'model': model,
                        'security_recall': comparative_analysis[strategy][model]['mean_security_recall']
                    })

        security_df = pd.DataFrame(security_data)
        if not security_df.empty:
            sns.barplot(data=security_df, x='model', y='security_recall', hue='strategy', ax=axes[1, 0])
            axes[1, 0].set_title('Recall کلاس‌های امنیتی')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Threat Detection Rate
        threat_data = []
        for strategy in strategies:
            for model in model_names:
                if model in comparative_analysis[strategy]:
                    threat_data.append({
                        'strategy': strategy,
                        'model': model,
                        'threat_detection': comparative_analysis[strategy][model]['threat_detection_rate']
                    })

        threat_df = pd.DataFrame(threat_data)
        if not threat_df.empty:
            sns.barplot(data=threat_df, x='model', y='threat_detection', hue='strategy', ax=axes[1, 1])
            axes[1, 1].set_title('نرخ شناسایی تهدید')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison_charts.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any],
                                y_true: pd.Series, strategy_name: str):
        """رسم و ذخیره ماتریس‌های سردرگمی برای تمام مدل‌ها"""
        output_dir = Path(self.config.MODEL_SAVE_PATHS['evaluation']) / "confusion_matrices"
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name, result in evaluation_results.items():
            if 'general_metrics' not in result:
                continue
            try:
                y_pred = np.array(result['predictions']['y_pred'])
                cm = confusion_matrix(y_true, y_pred)
                labels = np.unique(y_true)

                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {strategy_name} | {model_name}')
                plt.tight_layout()

                save_path = output_dir / f"cm_{strategy_name}_{model_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   💾 Confusion matrix saved: {save_path.name}")

            except Exception as e:
                print(f"⚠️  خطا در رسم ماتریس سردرگمی برای {model_name}: {e}")
