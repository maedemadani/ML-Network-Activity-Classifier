from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path


class DataCleaner:
    """کلاس عمومی برای پاکسازی داده‌های مختلف"""

    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        self.output_dir = output_dir or os.path.dirname(data_path)
        self.cleaning_report = {}
        self.df_original = None
        self.df_cleaned = None
        # تعریف متغیرهای تحلیل ستون‌ها
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.outlier_method = 'mark'
        self.allow_negative_cols = []
        self.illogical_values_removed = 0

    def load_data(self):
        """بارگذاری داده و بررسی اولیه - نسخه بهبود یافته"""
        print("=" * 50)
        print("گام 1: بارگذاری داده و بررسی اولیه")
        print("=" * 50)

        # تشخیص خودکار فرمت فایل
        file_extension = Path(self.data_path).suffix.lower()

        try:
            if file_extension == '.csv':
                df = pd.read_csv(self.data_path, low_memory=False)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(self.data_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(self.data_path)
            elif file_extension == '.json':
                df = pd.read_json(self.data_path)
            else:
                df = pd.read_csv(self.data_path, low_memory=False)
        except Exception as e:
            print(f"❌ خطا در خواندن فایل {self.data_path} با پسوند {file_extension}: {e}")
            raise

        # ذخیره کپی از داده اصلی
        self.df_original = df.copy()

        print(f"📁 فایل بارگذاری شد: {self.data_path}")
        print(f"📊 ابعاد داده: {df.shape} (ردیف‌ها: {df.shape[0]:,}, ستون‌ها: {df.shape[1]})")
        print(f"📝 فرمت فایل: {file_extension}")

        print("\n🔍 نمونه‌ای از داده‌ها:")
        print(df.head(3))

        print("\n📋 اطلاعات کلی داده‌ها:")
        df.info()

        # تحلیل اولیه ستون‌ها
        self._analyze_columns(df)

        # تشخیص خودکار انواع ستون‌ها
        self.column_analysis = self._auto_detect_column_types(df)

        # نمایش تحلیل ستون‌ها
        self._print_column_analysis()

        # ذخیره لیست ستون‌ها برای استفاده در متدهای دیگر
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        return df

    def _print_column_analysis(self):
        """چاپ تحلیل انواع ستون‌ها"""
        print(f"\n🎯 تحلیل خودکار ستون‌ها:")

        type_groups = {}
        for col, info in self.column_analysis.items():
            suggested_type = info['suggested_type']
            if suggested_type not in type_groups:
                type_groups[suggested_type] = []
            type_groups[suggested_type].append(col)

        for col_type, columns in type_groups.items():
            print(f"   {col_type}: {len(columns)} ستون")
            for col in columns[:5]:  # نمایش ۵ ستون اول
                unique_info = f" ({self.column_analysis[col]['unique_count']} مقدار یکتا)" if self.column_analysis[col][
                                                                                                  'unique_count'] < 100 else ""
                print(f"      • {col}{unique_info}")
            if len(columns) > 5:
                print(f"      • ... و {len(columns) - 5} ستون دیگر")

    def _analyze_columns(self, df):
        """تحلیل خودکار انواع ستون‌ها"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        print(f"\n🎯 تحلیل خودکار ستون‌ها:")
        print(f"   عددی: {len(numeric_cols)} ستون")
        print(f"   دسته‌ای: {len(categorical_cols)} ستون")
        print(f"   تاریخ/زمان: {len(datetime_cols)} ستون")

        self.cleaning_report['column_analysis'] = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }

    def remove_duplicates(self, df):
        """حذف ردیف‌های تکراری"""
        print("\n" + "=" * 50)
        print("گام 2: بررسی ردیف‌های تکراری")
        print("=" * 50)

        initial_count = len(df)
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            df = df.drop_duplicates()
            removed_count = initial_count - len(df)
            print(f"✅ {removed_count} ردیف تکراری حذف شد.")
            self.cleaning_report['duplicates_removed'] = removed_count
        else:
            print("✅ هیچ ردیف تکراری یافت نشد.")
            self.cleaning_report['duplicates_removed'] = 0

        print(f"📊 ابعاد جدید: {df.shape}")
        return df

    def handle_missing_values(self, df, strategy='auto'):
        """مدیریت مقادیر گمشده با گزارش دقیق"""
        print("\n" + "=" * 50)
        print("گام 3: مدیریت مقادیر گمشده")
        print("=" * 50)

        null_before = df.isnull().sum().sum()
        null_by_col_before = df.isnull().sum()

        if null_before == 0:
            print("✅ هیچ مقدار گمشده‌ای یافت نشد.")
            self.cleaning_report['missing_values'] = {'total_removed': 0}
            return df

        print(f"🔍 مقادیر گمشده قبل از پردازش: {null_before}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        fill_report = {}

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue

            null_percentage = (null_count / len(df)) * 100
            print(f"\n📊 {col}: {null_count} مقدار گمشده ({null_percentage:.1f}%)")

            if null_percentage > 50:  # حذف ستون اگر بیش از ۵۰٪ مقادیر گمشده
                df = df.drop(columns=[col])
                fill_report[col] = {'action': 'column_dropped', 'reason': 'high_missing_rate'}
                print(f"   🗑️  ستون حذف شد (نرخ گمشده بالا)")

            elif null_percentage > 15:  # حذف ردیف‌ها اگر ۱۵-۵۰٪ گمشده
                df = df.dropna(subset=[col])
                fill_report[col] = {'action': 'rows_dropped', 'count': null_count}
                print(f"   📝 {null_count} ردیف حذف شد")

            else:  # پر کردن مقادیر
                if col in numeric_cols:
                    fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    fill_report[col] = {'action': 'filled', 'value': fill_value, 'method': 'median'}
                    print(f"   🔢 پر شده با میانه: {fill_value}")
                elif col in categorical_cols:
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    fill_report[col] = {'action': 'filled', 'value': fill_value, 'method': 'mode'}
                    print(f"   📝 پر شده با مد: {fill_value}")
                else:
                    df[col] = df[col].fillna('Unknown')
                    fill_report[col] = {'action': 'filled', 'value': 'Unknown', 'method': 'constant'}
                    print(f"   🔤 پر شده با: Unknown")

        null_after = df.isnull().sum().sum()
        self.cleaning_report['missing_values'] = {
            'total_before': null_before,
            'total_after': null_after,
            'removed': null_before - null_after,
            'details': fill_report
        }

        print(f"\n✅ مقادیر گمشده پس از پردازش: {null_after}")
        return df

    def optimize_data_types(self, df):
        """بهینه‌سازی خودکار نوع داده‌ها """
        print("\n" + "=" * 50)
        print("گام 5: بهینه‌سازی نوع داده‌ها")
        print("=" * 50)

        conversion_report = {}

        for col, col_info in self.column_analysis.items():
            if col not in df.columns:
                continue

            original_dtype = str(df[col].dtype)
            suggested_type = col_info['suggested_type']

            try:
                if suggested_type == 'category' and not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype('category')
                    conversion_report[col] = {
                        'from': original_dtype,
                        'to': 'category',
                        'reason': 'low_cardinality'
                    }

                elif suggested_type == 'numeric' and pd.api.types.is_numeric_dtype(df[col]):
                    # بهینه‌سازی نوع عددی
                    if df[col].min() >= 0:
                        if df[col].max() < 255:
                            new_dtype = 'uint8'
                        elif df[col].max() < 65535:
                            new_dtype = 'uint16'
                        else:
                            new_dtype = 'uint32'
                    else:
                        if df[col].min() >= -128 and df[col].max() < 127:
                            new_dtype = 'int8'
                        elif df[col].min() >= -32768 and df[col].max() < 32767:
                            new_dtype = 'int16'
                        else:
                            new_dtype = 'int32'

                    df[col] = df[col].astype(new_dtype)
                    conversion_report[col] = {
                        'from': original_dtype,
                        'to': new_dtype,
                        'reason': 'numeric_optimization'
                    }

                elif suggested_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        conversion_report[col] = {
                            'from': original_dtype,
                            'to': 'datetime64[ns]',
                            'reason': 'temporal_conversion'
                        }
                    except:
                        pass

            except Exception as e:
                print(f"⚠️ خطا در بهینه‌سازی ستون {col}: {e}")

        # نمایش تغییرات
        if conversion_report:
            for col, info in conversion_report.items():
                print(f"✅ {col}: {info['from']} → {info['to']} ({info['reason']})")
        else:
            print("✅ همه نوع داده‌ها بهینه هستند")

        self.cleaning_report['type_conversions'] = conversion_report
        return df

    def validate_data_quality(self, df, allow_negative_cols=None):
        """اعتبارسنجی نهایی کیفیت داده"""
        print("\n" + "=" * 50)
        print("گام 6: اعتبارسنجی کیفیت داده")
        print("=" * 50)

        if allow_negative_cols is None:
            allow_negative_cols = []

        # بررسی مقادیر منفی فقط برای ستون‌های عددی که اجازه منفی ندارند
        numeric_cols_to_check = [col for col in self.numeric_cols if col not in allow_negative_cols]

        has_negative_numeric = False
        if numeric_cols_to_check:
            negative_check = df[numeric_cols_to_check] < 0
            has_negative_numeric = negative_check.any().any()

        validation_results = {
            'has_duplicates': df.duplicated().sum() == 0,
            'has_null_values': df.isnull().sum().sum() == 0,
            'data_types_optimized': len(self.cleaning_report.get('type_conversions', {})) > 0,
            'has_negative_numeric': has_negative_numeric,
        }

        # گزارش اعتبارسنجی
        checks = [
            ("عدم وجود تکراری‌ها", validation_results['has_duplicates'], "❌", "✅"),
            ("عدم وجود مقادیر Null", validation_results['has_null_values'], "❌", "✅"),
            ("عدم وجود مقادیر منفی غیرمنطقی", not validation_results['has_negative_numeric'], "⚠️", "✅"),
            ("نوع داده‌ها بهینه‌سازی شده", validation_results['data_types_optimized'], "ℹ️", "✅")
        ]

        for check_name, result, fail_icon, pass_icon in checks:
            icon = pass_icon if result else fail_icon
            print(f"{icon} {check_name}")

        all_passed = all([
            validation_results['has_duplicates'],
            validation_results['has_null_values'],
            not validation_results['has_negative_numeric']
        ])

        validation_results['all_passed'] = all_passed
        self.cleaning_report['validation'] = validation_results

        return all_passed

    def save_cleaned_data(self, df, output_path=None):
        """ذخیره داده پاک‌شده و گزارش """
        print("\n" + "=" * 50)
        print("گام 7: ذخیره نتایج")
        print("=" * 50)

        if output_path is None:
            input_name = Path(self.data_path).stem
            output_path = os.path.join(self.output_dir, f"{input_name}_cleaned.csv")

        # ایجاد پوشه خروجی اگر وجود ندارد
        os.makedirs(self.output_dir, exist_ok=True)

        # ذخیره داده پاک‌شده
        df.to_csv(output_path, index=False)
        print(f"💾 داده پاک‌شده ذخیره شد: {output_path}")

        # محاسبه آمار حافظه
        original_memory = self.df_original.memory_usage(deep=True).sum() if self.df_original is not None else 0
        cleaned_memory = df.memory_usage(deep=True).sum()
        memory_saved = original_memory - cleaned_memory
        memory_reduction_pct = (memory_saved / original_memory * 100) if original_memory > 0 else 0

        # ساخت گزارش با ساختار سلسله‌مراتبی
        comprehensive_report = {
            # ==================== METADATA ====================
            "metadata": {
                "project": "Network Activity Classification Pipeline",
                "phase": "Data Cleaning - Phase 1",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "input_file": str(self.data_path),
                "output_file": str(output_path),
                "cleaning_parameters": {
                    "outlier_method": getattr(self, 'outlier_method', 'mark'),
                    "allow_negative_cols": getattr(self, 'allow_negative_cols', [])
                },
                "execution_environment": {
                    "python_version": sys.version,
                    "pandas_version": pd.__version__,
                    "numpy_version": np.__version__
                }
            },

            # ==================== EXECUTION SUMMARY ====================
            "execution_summary": {
                "status": "completed_successfully",
                "original_dataset": {
                    "rows": int(self.df_original.shape[0]) if self.df_original is not None else 0,
                    "columns": int(self.df_original.shape[1]) if self.df_original is not None else 0,
                    "memory_bytes": int(original_memory)
                },
                "cleaned_dataset": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "memory_bytes": int(cleaned_memory)
                },
                "reduction_metrics": {
                    "rows_removed": int(
                        (self.df_original.shape[0] if self.df_original is not None else 0) - df.shape[0]),
                    "rows_removed_percentage": float((((self.df_original.shape[
                                                            0] if self.df_original is not None else 0) - df.shape[
                                                           0]) / (self.df_original.shape[
                                                                      0] if self.df_original is not None else 1)) * 100),
                    "memory_saved_bytes": int(memory_saved),
                    "memory_reduction_percentage": float(memory_reduction_pct)
                }
            },

            # ==================== DATA QUALITY ASSESSMENT ====================
            "data_quality_metrics": {
                "completeness_score": self._calculate_completeness_score(df),
                "consistency_score": self._calculate_consistency_score(df),
                "validity_score": self._calculate_validity_score(df),
                "uniqueness_score": self._calculate_uniqueness_score(df),
                "overall_quality_score": self._calculate_overall_quality_score(df),
                "quality_grades": {
                    "completeness": self._get_quality_grade(self._calculate_completeness_score(df)),
                    "consistency": self._get_quality_grade(self._calculate_consistency_score(df)),
                    "validity": self._get_quality_grade(self._calculate_validity_score(df)),
                    "overall": self._get_quality_grade(self._calculate_overall_quality_score(df))
                }
            },

            # ==================== CLEANING OPERATIONS DETAILS ====================
            "cleaning_operations": {
                "duplicate_removal": {
                    "rows_removed": self.cleaning_report.get('duplicates_removed', 0),
                    "remaining_duplicates": int(df.duplicated().sum()),
                    "effectiveness": "complete" if df.duplicated().sum() == 0 else "partial"
                },
                "missing_values_handling": self._enhance_missing_values_report(),
                "outlier_management": self._enhance_outlier_report(df),
                "data_type_optimization": {
                    "columns_optimized": len(self.cleaning_report.get('type_conversions', {})),
                    "conversions": self.cleaning_report.get('type_conversions', {}),
                    "memory_impact_bytes": int(memory_saved)
                },
                "invalid_data_removal": {
                    "rows_removed": getattr(self, 'illogical_values_removed', 0),
                    "types_found": ["negative_values", "out_of_range_ports"]
                }
            },

            # ==================== COLUMN-WISE ANALYSIS ====================
            "column_analysis": {
                "summary": {
                    "total_columns": len(df.columns),
                    "numeric_columns": len(self.numeric_cols),
                    "categorical_columns": len(self.categorical_cols),
                    "datetime_columns": len(self.datetime_cols),
                    "outlier_columns": len([col for col in df.columns if col.endswith('_outlier')])
                },
                "per_column_details": self._generate_detailed_column_analysis(df)
            },

            # ==================== STATISTICAL SUMMARY ====================
            "statistical_summary": {
                "dataset_statistics": {
                    "total_records": int(len(df)),
                    "total_columns": int(len(df.columns)),
                    "total_memory_bytes": int(cleaned_memory),
                    "average_record_size_bytes": int(cleaned_memory / len(df)) if len(df) > 0 else 0
                },
                "data_type_distribution": {
                    str(dtype): int(len(df.select_dtypes(include=[dtype]).columns))
                    for dtype in df.dtypes.unique()
                },
                "quality_indicators": {
                    "null_free_columns": int(len([col for col in df.columns if df[col].isnull().sum() == 0])),
                    "constant_columns": int(len([col for col in df.columns if df[col].nunique() <= 1])),
                    "high_cardinality_columns": int(
                        len([col for col in df.columns if df[col].nunique() > len(df) * 0.5]))
                }
            }
        }

        # ذخیره گزارش
        report_path = os.path.join(self.output_dir, "cleaning_report.json")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            columns_metadata = {
                "numeric_columns": self.numeric_cols,
                "categorical_columns": self.categorical_cols,
                "datetime_columns": self.datetime_cols,
                "all_columns": df.columns.tolist(),
                "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }

            columns_metadata_path = os.path.join(self.output_dir, "../columns_metadata.json")
            with open(columns_metadata_path, "w", encoding="utf-8") as f:
                json.dump(columns_metadata, f, indent=2, ensure_ascii=False)

            print(f"🧩 اطلاعات ستون‌ها ذخیره شد: {columns_metadata_path}")
            print(f"📊 گزارش حرفه‌ای پاکسازی ذخیره شد: {report_path}")

            # نمایش خلاصه‌ای از گزارش
            self._print_comprehensive_summary(comprehensive_report, df)

        except Exception as e:
            print(f"❌ خطا در ذخیره‌سازی گزارش: {e}")
            # Fallback: ذخیره گزارش ساده‌تر
            self._save_fallback_report(report_path, comprehensive_report)

        return output_path, report_path

    def _json_serializer(self, obj):
        """تابع سریالایزر حرفه‌ای برای JSON"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'isoformat'):  # برای datetime
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Categorical):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _calculate_completeness_score(self, df):
        """محاسبه نمره کامل بودن داده‌ها"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 1.0
        return float(completeness * 100)

    def _calculate_consistency_score(self, df):
        """محاسبه نمره سازگاری داده‌ها """
        try:
            consistency_checks = []

            # بررسی Bytes consistency
            if all(col in df.columns for col in ['Bytes', 'Bytes Sent', 'Bytes Received']):
                # استفاده از np.isclose برای مقایسه اعداد اعشاری
                bytes_consistent = np.isclose(
                    df['Bytes'],
                    df['Bytes Sent'] + df['Bytes Received'],
                    rtol=1e-5
                ).mean()
                consistency_checks.append(float(bytes_consistent))

            # بررسی Packets consistency
            if all(col in df.columns for col in ['Packets', 'pkts_sent', 'pkts_received']):
                packets_consistent = np.isclose(
                    df['Packets'],
                    df['pkts_sent'] + df['pkts_received'],
                    rtol=1e-5
                ).mean()
                consistency_checks.append(float(packets_consistent))

            return float(np.mean(consistency_checks) * 100) if consistency_checks else 100.0

        except Exception as e:
            print(f"⚠️ خطا در محاسبه سازگاری: {e}")
            return 100.0  # در صورت خطا

    def _calculate_validity_score(self, df):
        """محاسبه نمره اعتبار داده‌ها """
        try:
            valid_cells = 0
            total_cells = df.shape[0] * df.shape[1]

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # بررسی مقادیر منفی فقط برای ستون‌هایی که نباید منفی باشند
                    if col not in getattr(self, 'allow_negative_cols', []):
                        valid_cells += (df[col] >= 0).sum()
                    else:
                        valid_cells += len(df)  # ستون‌های مجاز برای منفی
                else:
                    valid_cells += len(df)  # برای سایر انواع

            return float((valid_cells / total_cells) * 100) if total_cells > 0 else 100.0
        except Exception as e:
            print(f"⚠️ خطا در محاسبه اعتبار: {e}")
            return 100.0

    def _calculate_uniqueness_score(self, df):
        """محاسبه نمره یکتایی داده‌ها"""
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        return float((1 - duplicate_ratio) * 100)

    def _calculate_overall_quality_score(self, df):
        """محاسبه نمره کلی کیفیت داده"""
        scores = [
            self._calculate_completeness_score(df) * 0.3,
            self._calculate_consistency_score(df) * 0.25,
            self._calculate_validity_score(df) * 0.25,
            self._calculate_uniqueness_score(df) * 0.2
        ]
        return float(sum(scores))

    def _get_quality_grade(self, score):
        """درجه‌بندی کیفیت بر اساس نمره"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        else:
            return "D"

    def _enhance_missing_values_report(self):
        """گزارش پیشرفته مدیریت مقادیر گمشده"""
        missing_report = self.cleaning_report.get('missing_values', {})

        enhanced_report = {
            "total_missing_before": missing_report.get('total_before', 0),
            "total_missing_after": missing_report.get('total_after', 0),
            "missing_resolution_rate": float(
                (missing_report.get('total_before', 0) - missing_report.get('total_after', 0)) /
                missing_report.get('total_before', 1) * 100
            ) if missing_report.get('total_before', 0) > 0 else 100.0,
            "handling_strategy": "auto_imputation",
            "details": missing_report.get('details', {})
        }

        return enhanced_report

    def _enhance_outlier_report(self, df):
        """گزارش پیشرفته مدیریت outliers"""
        outlier_report = self.cleaning_report.get('outliers', {})

        total_outliers = sum(stats.get('count', 0) for stats in outlier_report.values())
        total_cells = df.shape[0] * len([col for col in outlier_report.keys() if not col.endswith('_outlier')])

        enhanced_report = {
            "total_columns_analyzed": len(outlier_report),
            "total_outliers_detected": total_outliers,
            "outlier_percentage": float((total_outliers / total_cells) * 100) if total_cells > 0 else 0,
            "outlier_management_strategy": getattr(self, 'outlier_method', 'mark'),
            "columns_with_outliers": [
                {
                    "column": col,
                    "outlier_count": stats.get('count', 0),
                    "outlier_percentage": stats.get('percentage', 0),
                    "normal_range": [stats.get('lower_bound', 0), stats.get('upper_bound', 0)],
                    "actual_range": [stats.get('min', 0), stats.get('max', 0)]
                }
                for col, stats in outlier_report.items() if not col.endswith('_outlier')
            ]
        }

        return enhanced_report

    def _generate_detailed_column_analysis(self, df):
        """تولید تحلیل دقیق برای هر ستون"""
        column_details = {}

        for col in df.columns:
            col_info = {
                "data_type": str(df[col].dtype),
                "total_values": int(len(df)),
                "missing_values": {
                    "count": int(df[col].isnull().sum()),
                    "percentage": float((df[col].isnull().sum() / len(df)) * 100)
                },
                "unique_values": {
                    "count": int(df[col].nunique()),
                    "percentage": float((df[col].nunique() / len(df)) * 100)
                },
                "data_category": self._get_data_category(df[col])
            }

            # آمارهای عددی
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["numerical_statistics"] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "q1": float(df[col].quantile(0.25)),
                    "q3": float(df[col].quantile(0.75))
                }

            # آمارهای دسته‌ای
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts()
                col_info["categorical_statistics"] = {
                    "top_values": {
                        str(value): int(count) for value, count in value_counts.head(10).items()
                    },
                    "value_distribution": {
                        "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        "most_frequent_percentage": float((value_counts.iloc[0] / len(df)) * 100) if len(
                            value_counts) > 0 else 0
                    }
                }

            column_details[col] = col_info

        return column_details

    def _get_data_category(self, series):
        """تعیین دسته داده برای ستون"""
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"
        elif pd.api.types.is_string_dtype(series):
            return "text"
        elif pd.api.types.is_categorical_dtype(series):
            return "categorical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "unknown"

    def _print_comprehensive_summary(self, report, df):
        """چاپ خلاصه جامع از گزارش"""
        print(f"\n📋 خلاصه جامع گزارش پاکسازی:")
        print(
            f"   📊 ابعاد داده: {report['execution_summary']['cleaned_dataset']['rows']:,} سطر × {report['execution_summary']['cleaned_dataset']['columns']} ستون")
        print(
            f"   💾 صرفه‌جویی حافظه: {report['execution_summary']['reduction_metrics']['memory_saved_bytes']:,} بایت ({report['execution_summary']['reduction_metrics']['memory_reduction_percentage']:.1f}%)")
        print(
            f"   🧹 سطرهای حذف شده: {report['execution_summary']['reduction_metrics']['rows_removed']:,} ({report['execution_summary']['reduction_metrics']['rows_removed_percentage']:.1f}%)")
        print(
            f"   📈 نمره کیفیت داده: {report['data_quality_metrics']['overall_quality_score']:.1f}% ({report['data_quality_metrics']['quality_grades']['overall']})")
        print(
            f"   🎯 Outliers شناسایی شده: {report['cleaning_operations']['outlier_management']['total_outliers_detected']:,}")
        print(
            f"   🔧 ستون‌های بهینه‌شده: {report['cleaning_operations']['data_type_optimization']['columns_optimized']}")

    def _save_fallback_report(self, report_path, comprehensive_report):
        """ذخیره گزارش در صورت بروز خطا"""
        fallback_report = {
            "metadata": comprehensive_report["metadata"],
            "execution_summary": comprehensive_report["execution_summary"],
            "status": "completed_with_warnings",
            "error": "Simplified report due to serialization issues"
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(fallback_report, f, indent=2, ensure_ascii=False)
        print(f"📊 گزارش ساده ذخیره شد: {report_path}")

    def detect_and_handle_outliers(self, df, method='remove'):
        """شناسایی و مدیریت outlierها با قابلیت حذف"""
        print("\n" + "=" * 50)
        print("گام 4: شناسایی و مدیریت Outliers")
        print("=" * 50)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}

        initial_row_count = len(df)
        outlier_flags = pd.Series([False] * len(df), index=df.index)

        # محاسبه فراوانی هر برچسب برای تشخیص کلاس‌های کم‌داده
        target_col = "Action" if "Action" in df.columns else None
        rare_classes = set()
        if target_col:
            class_counts = df[target_col].value_counts()
            rare_classes = set(class_counts[class_counts < 100].index)  # <100 نمونه یعنی کلاس کم‌تعداد
            if rare_classes:
                print(f"🔎 کلاس‌های کم‌تعداد شناسایی‌شده و محافظت‌شده در حذف Outlierها: {rare_classes}")

        for col in numeric_cols:
            try:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                outliers_mask = (df[col] < lower) | (df[col] > upper)
                outlier_count = outliers_mask.sum()
                outlier_percentage = (outlier_count / len(df)) * 100

                outlier_report[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'min': df[col].min(),
                    'max': df[col].max()
                }

                print(f"\n📊 {col}:")
                print(f"   📈 محدوده معمول: [{lower:.2f}, {upper:.2f}]")
                print(f"   📊 مقادیر واقعی: [{df[col].min():.2f}, {df[col].max():.2f}]")
                print(f"   ⚠️  تعداد outliers: {outlier_count} ({outlier_percentage:.1f}%)")

                outlier_flags = outlier_flags | outliers_mask

            except Exception as e:
                print(f"⚠️ خطا در پردازش outliers برای ستون {col}: {e}")

        # 🚫 جلوگیری از حذف داده‌های کم‌تعداد
        if method == 'remove' and outlier_flags.any():
            protected_mask = pd.Series(False, index=df.index)
            if target_col and rare_classes:
                protected_mask = df[target_col].isin(rare_classes)
            combined_mask = outlier_flags & ~protected_mask

            df_clean = df[~combined_mask]
            removed_count = len(df) - len(df_clean)
            protected_count = (outlier_flags & protected_mask).sum()

            print(f"\n🗑️  حذف {removed_count} ردیف outlier ({removed_count / len(df) * 100:.1f}%)")
            if protected_count:
                print(f"🛡️  حفظ {protected_count} outlier از کلاس‌های کم‌تعداد ({rare_classes})")

            self.cleaning_report['outliers_removed'] = removed_count
            df = df_clean

        self.cleaning_report['outliers'] = outlier_report
        return df

    def clean(self, outlier_method='remove', allow_negative_cols=None):
        """اجرای کامل فرآیند پاکسازی"""
        print("🚀 شروع فرآیند پاکسازی داده‌ها")
        print("=" * 60)

        try:
            # ذخیره پارامترها
            self.outlier_method = outlier_method
            self.allow_negative_cols = allow_negative_cols or []

            # اجرای مراحل پاکسازی
            df = self.load_data()
            df = self.remove_duplicates(df)
            df = self.handle_missing_values(df)
            df = self.detect_and_handle_outliers(df, method=outlier_method)
            df = self.optimize_data_types(df)

            # اعتبارسنجی نهایی
            is_valid = self.validate_data_quality(df, allow_negative_cols=allow_negative_cols)
            self.df_cleaned = df

            # ذخیره نتایج
            data_path, report_path = self.save_cleaned_data(df)

            return {
                'success': True,
                'cleaned_data': df,
                'data_path': data_path,
                'report_path': report_path,
                'is_valid': is_valid,
                'report': self.cleaning_report
            }

        except Exception as e:
            print(f"❌ خطا در پاکسازی: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _auto_detect_column_types(self, df):
        """تشخیص خودکار انواع ستون‌ها بر اساس نام و محتوا"""
        column_analysis = {}

        for col in df.columns:
            col_info = {
                'original_dtype': str(df[col].dtype),
                'unique_count': df[col].nunique(),
                'null_count': df[col].isnull().sum(),
                'suggested_type': None,
                'is_identifier': False,
                'is_temporal': False,
                'is_categorical': False
            }

            # تشخیص بر اساس نام ستون
            col_lower = col.lower()

            if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'index']):
                col_info['is_identifier'] = True
                col_info['suggested_type'] = 'identifier'

            elif any(keyword in col_lower for keyword in ['time', 'date', 'year', 'month', 'day']):
                col_info['is_temporal'] = True
                if pd.api.types.is_string_dtype(df[col]):
                    # سعی در تبدیل به datetime
                    try:
                        pd.to_datetime(df[col], errors='coerce')
                        col_info['suggested_type'] = 'datetime'
                    except:
                        col_info['suggested_type'] = 'string'

            elif any(keyword in col_lower for keyword in ['port', 'protocol', 'type', 'status', 'action']):
                col_info['is_categorical'] = True
                if df[col].nunique() < 50:  # تعداد دسته‌های کم
                    col_info['suggested_type'] = 'category'
                else:
                    col_info['suggested_type'] = 'numeric'

            elif any(keyword in col_lower for keyword in ['byte', 'packet', 'size', 'count', 'length']):
                col_info['suggested_type'] = 'numeric'

            elif any(keyword in col_lower for keyword in ['ratio', 'rate', 'percent', 'probability']):
                col_info['suggested_type'] = 'numeric_float'

            else:
                # تشخیص بر اساس داده
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info['suggested_type'] = 'numeric'
                elif df[col].nunique() / len(df) < 0.1:  # کمتر از 10% مقادیر یکتا
                    col_info['suggested_type'] = 'category'
                else:
                    col_info['suggested_type'] = 'string'

            column_analysis[col] = col_info

        return column_analysis


# تابع main برای سازگاری با کد موجود
def main(data_path=None, output_dir=None, allow_negative_cols=None, outlier_method='mark'):
    """تابع اصلی برای اجرای پاکسازی - سازگار با main.py اصلی"""
    print("=" * 60)
    print("فاز 1: پاکسازی و آماده‌سازی داده‌ها")
    print("=" * 60)

    # استفاده از مسیر پیش‌فرض اگر داده‌ای ارائه نشده
    if data_path is None:
        data_path = "../data/network_logs.csv"

    # اگر مسیر نسبی است، آن را به مسیر کامل تبدیل کن
    if not os.path.isabs(data_path):
        # فرض می‌کنیم فایل در پوشه data قرار دارد
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / data_path

    # اگر output_dir مشخص نشده، از پوشه data استفاده کن
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data/cleanedData"

    print(f"📁 فایل ورودی: {data_path}")
    print(f"📂 پوشه خروجی: {output_dir}")
    print(f"🎯 روش مدیریت outliers: {outlier_method}")
    if allow_negative_cols:
        print(f"✅ ستون‌های مجاز برای مقادیر منفی: {allow_negative_cols}")

    try:
        # ایجاد نمونه DataCleaner و اجرای پاکسازی
        cleaner = DataCleaner(str(data_path), str(output_dir))
        result = cleaner.clean(
            outlier_method=outlier_method,
            allow_negative_cols=allow_negative_cols
        )

        # گزارش نتیجه
        if result['success']:
            print("\n" + "=" * 50)
            print("✅ فاز 1 با موفقیت به پایان رسید!")
            print(f"📊 داده پاک‌شده: {result['data_path']}")
            print(f"📈 گزارش: {result['report_path']}")
            print("=" * 50)
        else:
            print(f"\n❌ فاز 1 با خطا مواجه شد: {result.get('error', 'خطای ناشناخته')}")

        return result

    except Exception as e:
        error_result = {
            'success': False,
            'error': f"خطا در اجرای فاز 1: {str(e)}"
        }
        print(f"❌ {error_result['error']}")
        return error_result


if __name__ == "__main__":
    # پشتیبانی از آرگومان‌های خط فرمان (برای اجرای مستقل)
    import argparse

    parser = argparse.ArgumentParser(description='پاک‌کننده عمومی داده‌ها - فاز 1')
    parser.add_argument('--input', type=str, help='مسیر فایل ورودی', default='network_logs.csv')
    parser.add_argument('--output', type=str, help='پوشه خروجی', default=None)
    parser.add_argument('--outlier-method', choices=['remove', 'mark', 'clip', 'ignore'],
                        default='remove', help='روش مدیریت outliers (پیش‌فرض: remove)')
    # parser.add_argument('--allow-negative', nargs='*', help='ستون‌های مجاز برای مقادیر منفی', default=[])

    args = parser.parse_args()

    # فراخوانی تابع main با آرگومان‌های خط فرمان
    main(
        data_path=args.input,
        output_dir=args.output,
        outlier_method=args.outlier_method,
        # allow_negative_cols=args.allow_negative
    )