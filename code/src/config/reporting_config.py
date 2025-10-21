from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


@dataclass
class Phase5Config:
    """پیکربندی جامع فاز ۵ - تحلیل نهایی و تحویل"""

    # تنظیمات مسیرها
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    RUN_PREFIX: str = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # تنظیمات آنالیز
    STATISTICAL_ALPHA: float = 0.05
    N_BOOTSTRAP_SAMPLES: int = 1000
    TOP_FEATURES_COUNT: int = 10

    # معیارهای انتخاب مدل
    SELECTION_CRITERIA: Dict[str, float] = None
    MINORITY_CLASSES: List[int] = None

    # تنظیمات گزارش‌دهی
    REPORT_FORMATS: List[str] = None

    def __post_init__(self):
        if self.SELECTION_CRITERIA is None:
            self.SELECTION_CRITERIA = {
                'f1_minority_mean': 0.6,
                'min_recall_minority': 0.7,
                'max_inference_time_ms': 50,
                'max_model_size_mb': 100
            }

        if self.MINORITY_CLASSES is None:
            self.MINORITY_CLASSES = [1, 2]  # deny, drop

        if self.REPORT_FORMATS is None:
            self.REPORT_FORMATS = ['pdf', 'html', 'ipynb']

    def get_run_directory(self) -> Path:
        """دریافت مسیر پوشه اجرای فعلی"""
        return self.BASE_DIR / "final_report" / self.RUN_PREFIX

    def get_subdirectories(self) -> Dict[str, Path]:
        """دریافت مسیرهای تمام زیرپوشه‌ها"""
        run_dir = self.get_run_directory()
        return {
            'models': run_dir / "models",
            'reports': run_dir / "reports",
            'plots': run_dir / "plots",
            'tables': run_dir / "tables",
            'notebooks': run_dir / "notebooks",
            'metadata': run_dir / "metadata",
            'runbook': run_dir / "runbook"
        }