import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np
import subprocess
import os


class RunManager:
    """مدیریت اجرا و متادیتای فاز ۵ """

    def __init__(self, config):
        self.config = config
        self.run_metadata = {}
        self.subdirectories = config.get_subdirectories()

    def setup_run_environment(self) -> bool:
        """ایجاد محیط اجرا و پوشه‌های مورد نیاز"""
        try:
            # ایجاد پوشه‌ها
            for dir_name, dir_path in self.subdirectories.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ پوشه ایجاد شد: {dir_path}")

            # جمع‌آوری متادیتا
            self._collect_metadata()

            # ذخیره متادیتا
            metadata_path = self.subdirectories['metadata'] / "run_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.run_metadata, f, indent=2, ensure_ascii=False)

            print(f"🏁 محیط اجرا در {self.config.get_run_directory()} آماده شد")
            return True

        except Exception as e:
            print(f"❌ خطا در ایجاد محیط اجرا: {e}")
            return False

    def _collect_metadata(self):
        """جمع‌آوری متادیتای اجرا"""
        self.run_metadata = {
            'run_id': self.config.RUN_PREFIX,
            'timestamp': datetime.now().isoformat(),
            'phase': 'Final Analysis - Phase 5',
            'git_info': self._get_git_info_safe(),
            'config': {
                'statistical_alpha': self.config.STATISTICAL_ALPHA,
                'bootstrap_samples': self.config.N_BOOTSTRAP_SAMPLES,
                'selection_criteria': self.config.SELECTION_CRITERIA,
                'minority_classes': self.config.MINORITY_CLASSES
            },
            'system_info': self._get_system_info()
        }

    def _get_git_info_safe(self) -> Dict[str, Any]:
        """دریافت اطلاعات Git به صورت ایمن (بدون وابستگی به GitPython)"""
        try:
            # استفاده از دستورات git از طریق subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            commit_hash = result.stdout.strip()

            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, check=True
            )
            branch = result.stdout.strip()

            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, check=True
            )
            is_dirty = len(result.stdout.strip()) > 0

            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=%B'],
                capture_output=True, text=True, check=True
            )
            commit_message = result.stdout.strip()

            return {
                'commit_hash': commit_hash,
                'branch': branch,
                'is_dirty': is_dirty,
                'commit_message': commit_message
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            # اگر git موجود نبود یا خطایی رخ داد
            return {
                'available': False,
                'error': 'Git not available or not a git repository'
            }

    def _get_system_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات سیستم"""
        import platform
        import sys
        return {
            'python_version': platform.python_version(),
            'system': platform.system(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'executable': sys.executable
        }

    def copy_phase4_artifacts(self, phase4_path: Path) -> bool:
        """کپی آرتیفکت‌های فاز ۴"""
        try:
            # مسیرهای مختلفی که ممکن است آرتیفکت‌ها وجود داشته باشند
            possible_paths = [
                phase4_path / "data" / "models",
                Path("../data/models"),
                Path("data/models")
            ]

            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break

            if source_path is None:
                print("⚠️  آرتیفکت‌های فاز ۴ یافت نشدند. مسیرهای بررسی شده:")
                for path in possible_paths:
                    print(f"   • {path}")
                return False

            # کپی مدل‌ها
            dest_models = self.subdirectories['models']
            if (source_path / "trained_models").exists():
                shutil.copytree(source_path / "trained_models", dest_models / "trained_models")

            if (source_path / "tuned_models").exists():
                shutil.copytree(source_path / "tuned_models", dest_models / "tuned_models")

            if (source_path / "preprocessors").exists():
                shutil.copytree(source_path / "preprocessors", dest_models / "preprocessors")

            #  کپی نتایج ارزیابی با تشخیص خودکار مسیر
            eval_candidates = [
                source_path / "evaluation",
                source_path / "evaluation_results",
                source_path / "data" / "models" / "evaluation",
                source_path / "data" / "models" / "evaluation_results"
            ]

            found_eval = False
            for eval_path in eval_candidates:
                if eval_path.exists():
                    dest_eval = self.subdirectories['tables'] / "evaluation_results"
                    shutil.copytree(eval_path, dest_eval, dirs_exist_ok=True)
                    print(f"✅ نتایج ارزیابی از {eval_path} کپی شد → {dest_eval}")
                    found_eval = True
                    break

            if not found_eval:
                print("⚠️  پوشه نتایج ارزیابی یافت نشد در مسیرهای ممکن:")
                for candidate in eval_candidates:
                    print(f"   • {candidate}")

            print("✅ آرتیفکت‌های فاز ۴ کپی شدند")
            return True

        except Exception as e:
            print(f"❌ خطا در کپی آرتیفکت‌ها: {e}")
            return False