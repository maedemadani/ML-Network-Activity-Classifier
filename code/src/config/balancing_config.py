from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class BalanceConfig:
    """پیکربندی عمومی برای مدیریت عدم تعادل کلاس‌ها"""

    # پارامترهای تقسیم داده
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # آستانه‌های کلاس نادر
    RARE_CLASS_THRESHOLD: float = 0.005  # 0.5%
    MIN_SAMPLES_FOR_SMOTE: int = 2

    # استراتژی‌های نمونه‌برداری
    SAMPLING_STRATEGIES: Dict[str, Any] = None

    # کلاس‌های معادل برای ادغام
    CLASS_MERGE_MAPPING: Dict[str, str] = None

    # نام‌های احتمالی ستون هدف
    TARGET_COLUMN_NAMES: List[str] = None

    def __post_init__(self):
        if self.SAMPLING_STRATEGIES is None:
            self.SAMPLING_STRATEGIES = {
                'undersampling': {
                    'strategy': 'auto',  # 'auto', 'median', 'majority'
                    'random_state': self.RANDOM_STATE
                },
                'oversampling': {
                    'strategy': 'auto',
                    'k_neighbors': 'adaptive',
                    'random_state': self.RANDOM_STATE
                }
            }

        if self.CLASS_MERGE_MAPPING is None:
            self.CLASS_MERGE_MAPPING = {
                'reset-both': 'drop',
                'reset_client': 'drop',
                'reset_server': 'drop'
            }

        if self.TARGET_COLUMN_NAMES is None:
            self.TARGET_COLUMN_NAMES = ['Action', 'Class', 'Label', 'Target', 'category']