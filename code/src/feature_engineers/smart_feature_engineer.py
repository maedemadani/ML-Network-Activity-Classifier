import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class SmartFeatureEngineer:
    """کلاس هوشمند مهندسی ویژگی برای دیتاست‌های شبکه"""

    def __init__(self, metadata_path, target_column='Action', scaling_strategy='standard'):
        self.metadata = self._load_metadata(metadata_path)
        self.target_column = target_column
        self.scaling_strategy = scaling_strategy
        self.detected_features = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.fitted = False

    def _load_metadata(self, metadata_path):
        """بارگذاری متادیتای ستون‌ها از فاز ۱"""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"خطا در بارگذاری متادیتا: {e}")

    def _find_columns_by_pattern(self, patterns):
        """پیدا کردن ستون‌ها بر اساس الگوهای نام - عمومی برای هر دیتاست"""
        found_columns = []
        for col in self.metadata['all_columns']:
            if col == self.target_column:
                continue
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    found_columns.append(col)
                    break
        return list(set(found_columns))  # حذف موارد تکراری

    def detect_available_features(self):
        """تشخیص خودکار انواع ویژگی‌های موجود در dataset"""
        print("🔍 در حال تشخیص خودکار ویژگی‌های موجود...")

        self.detected_features = {
            # پورت‌ها
            'source_ports': self._find_columns_by_pattern(['source port', 'src_port', 'src port']),
            'destination_ports': self._find_columns_by_pattern(['destination port', 'dst_port', 'dst port']),
            'nat_ports': self._find_columns_by_pattern(['nat source', 'nat destination', 'nat_port']),

            # ترافیک
            'bytes_columns': self._find_columns_by_pattern(['bytes', 'byte']),
            'packets_columns': self._find_columns_by_pattern(['packets', 'packet']),
            'sent_columns': self._find_columns_by_pattern(['sent', 'forward']),
            'received_columns': self._find_columns_by_pattern(['received', 'backward']),

            # زمان
            'time_columns': self._find_columns_by_pattern(['time', 'duration', 'elapsed', 'interval']),
            'timestamp_columns': self._find_columns_by_pattern(['timestamp', 'date', 'time']),
        }

        # گزارش تشخیص
        print("✅ ویژگی‌های تشخیص داده شده:")
        for feature_type, columns in self.detected_features.items():
            if columns:
                print(f"   📊 {feature_type}: {columns}")

        return self.detected_features

    def fit_transform(self, df, train_mask=None):
        """آموزش و تبدیل داده‌ها - برای Train set"""
        print("🚀 شروع فرآیند مهندسی ویژگی‌ها ")
        print("=" * 60)

        self.fitted = True
        df_processed = df.copy()

        # تشخیص ویژگی‌ها
        self.detect_available_features()

        # ۱. ایجاد ویژگی‌های NAT
        df_processed = self._create_nat_features(df_processed)

        # ۲. مهندسی ویژگی‌های پورتی
        df_processed = self._engineer_port_features(df_processed)

        # ۳. مهندسی ویژگی‌های ترافیکی
        df_processed = self._engineer_traffic_features(df_processed)

        # ۴. مهندسی ویژگی‌های زمانی
        df_processed = self._engineer_time_features(df_processed)

        # ۵. تبدیل لگاریتمی
        df_processed = self._apply_log_transforms(df_processed)

        # ۶. رمزگذاری ویژگی‌های دسته‌ای
        df_processed = self._encode_categorical_features(df_processed)

        # ۷. مقیاس‌بندی (فقط روی train)
        if train_mask is not None:
            df_train = df_processed[train_mask].copy()
            df_train = self._scale_features(df_train, fit_scalers=True)
            df_processed.loc[train_mask] = df_train
        else:
            df_processed = self._scale_features(df_processed, fit_scalers=True)

        # ۸. انتخاب ویژگی
        df_processed = self._select_features(df_processed)

        print(f"✅ مهندسی ویژگی‌ها completed! ابعاد نهایی: {df_processed.shape}")
        return df_processed

    def transform(self, df):
        """تبدیل داده‌های جدید - برای Test set"""
        if not self.fitted:
            raise Exception("ابتدا باید متد fit_transform فراخوانی شود")

        print("🔧 تبدیل داده‌های جدید - حالت آزمون")
        df_processed = df.copy()

        # ۱. ایجاد ویژگی‌های NAT
        df_processed = self._create_nat_features(df_processed)

        # ۲. مهندسی ویژگی‌های پورتی
        df_processed = self._engineer_port_features(df_processed)

        # ۳. مهندسی ویژگی‌های ترافیکی
        df_processed = self._engineer_traffic_features(df_processed)

        # ۴. مهندسی ویژگی‌های زمانی
        df_processed = self._engineer_time_features(df_processed)

        # ۵. تبدیل لگاریتمی
        df_processed = self._apply_log_transforms(df_processed)

        # ۶. رمزگذاری ویژگی‌های دسته‌ای
        df_processed = self._encode_categorical_features(df_processed, is_train=False)

        # ۷. مقیاس‌بندی (با scalerهای آموزش دیده)
        df_processed = self._scale_features(df_processed, fit_scalers=False)

        # ۸. انتخاب ویژگی (با selector آموزش دیده)
        df_processed = self._select_features(df_processed, is_train=False)

        return df_processed

    def _create_nat_features(self, df):
        """ایجاد ویژگی‌های NAT - عمومی"""
        print("🔧 ایجاد ویژگی‌های NAT...")

        nat_features_added = 0

        # پرچم NAT کلی
        nat_port_cols = self.detected_features['nat_ports']
        if nat_port_cols:
            # بررسی آیا هر کدام از ستون‌های NAT مقدار غیرصفر دارند
            nat_flags = df[nat_port_cols].fillna(0) != 0
            df['has_nat'] = nat_flags.any(axis=1).astype(int)
            nat_features_added += 1

            # پرچم‌های تطابق پورت‌های NAT با اصلی
            src_port_cols = self.detected_features['source_ports']
            dst_port_cols = self.detected_features['destination_ports']

            if src_port_cols and len(nat_port_cols) >= 1:
                src_col = src_port_cols[0]
                nat_src_col = [col for col in nat_port_cols if 'source' in col.lower() or 'src' in col.lower()]
                if nat_src_col:
                    df['src_port_nat_match'] = (df[src_col] == df[nat_src_col[0]]).astype(int)
                    nat_features_added += 1

            if dst_port_cols and len(nat_port_cols) >= 1:
                dst_col = dst_port_cols[0]
                nat_dst_col = [col for col in nat_port_cols if 'destination' in col.lower() or 'dst' in col.lower()]
                if nat_dst_col:
                    df['dst_port_nat_match'] = (df[dst_col] == df[nat_dst_col[0]]).astype(int)
                    nat_features_added += 1

        print(f"   ✅ {nat_features_added} ویژگی NAT ایجاد شد")
        return df

    def _engineer_port_features(self, df):
        """مهندسی ویژگی‌های پورتی - عمومی"""
        print("🔧 ایجاد ویژگی‌های پورتی...")

        port_features_added = 0

        # دسته‌بندی پورت مبدا
        src_port_cols = self.detected_features['source_ports']
        if src_port_cols:
            for col in src_port_cols:
                df[f'{col}_category'] = df[col].apply(self._categorize_port)
                port_features_added += 1

                # پرچم‌های سرویس مهم
                df[f'{col}_is_well_known'] = (df[col] <= 1023).astype(int)
                df[f'{col}_is_ephemeral'] = (df[col] >= 49152).astype(int)
                port_features_added += 2

        # دسته‌بندی پورت مقصد
        dst_port_cols = self.detected_features['destination_ports']
        if dst_port_cols:
            for col in dst_port_cols:
                df[f'{col}_category'] = df[col].apply(self._categorize_port)
                port_features_added += 1

                # شناسایی سرویس‌های رایج
                df[f'{col}_is_http'] = (df[col].isin([80, 8080, 443])).astype(int)
                df[f'{col}_is_dns'] = (df[col] == 53).astype(int)
                df[f'{col}_is_ssh'] = (df[col] == 22).astype(int)
                port_features_added += 3

        print(f"   ✅ {port_features_added} ویژگی پورتی ایجاد شد")
        return df

    def _categorize_port(self, port):
        """دسته‌بندی پورت - عمومی"""
        try:
            port = int(port)
            if port <= 1023:
                return 'well_known'
            elif port <= 49151:
                return 'registered'
            else:
                return 'ephemeral'
        except:
            return 'unknown'

    def _engineer_traffic_features(self, df):
        """مهندسی ویژگی‌های ترافیکی - عمومی"""
        print("🔧 ایجاد ویژگی‌های ترافیکی...")

        traffic_features_added = 0
        epsilon = 1e-8

        # نسبت‌های ارسال/دریافت
        bytes_cols = self.detected_features['bytes_columns']
        sent_cols = self.detected_features['sent_columns']
        received_cols = self.detected_features['received_columns']
        packets_cols = self.detected_features['packets_columns']

        # نسبت بایت‌های ارسالی
        if len(bytes_cols) >= 1 and len(sent_cols) >= 1:
            total_bytes_col = bytes_cols[0]
            sent_bytes_col = sent_cols[0]
            df['bytes_sent_ratio'] = df[sent_bytes_col] / (df[total_bytes_col] + epsilon)
            traffic_features_added += 1

        # نسبت پکت‌های ارسالی
        if len(packets_cols) >= 1 and len(sent_cols) >= 1:
            # پیدا کردن ستون pkts_sent
            pkts_sent_cols = self._find_columns_by_pattern(['pkts_sent', 'packets_sent'])
            pkts_received_cols = self._find_columns_by_pattern(['pkts_received', 'packets_received'])

            if pkts_sent_cols and pkts_received_cols:
                df['pkts_sent_ratio'] = df[pkts_sent_cols[0]] / (
                            df[pkts_sent_cols[0]] + df[pkts_received_cols[0]] + epsilon)
                traffic_features_added += 1

        # کارایی ترافیک
        if len(bytes_cols) >= 1 and len(packets_cols) >= 1:
            df['avg_packet_size'] = df[bytes_cols[0]] / (df[packets_cols[0]] + epsilon)
            traffic_features_added += 1

        # پرچم‌های ترافیک غیرعادی
        if len(bytes_cols) >= 1:
            df['is_small_flow'] = (df[bytes_cols[0]] < 100).astype(int)
            df['is_large_flow'] = (df[bytes_cols[0]] > 1000000).astype(int)
            traffic_features_added += 2

        print(f"   ✅ {traffic_features_added} ویژگی ترافیکی ایجاد شد")
        return df

    def _engineer_time_features(self, df):
        """مهندسی ویژگی‌های زمانی - عمومی"""
        print("🔧 ایجاد ویژگی‌های زمانی...")

        time_features_added = 0
        epsilon = 1e-8

        time_cols = self.detected_features['time_columns']
        timestamp_cols = self.detected_features['timestamp_columns']

        # نرخ‌های مبتنی بر زمان
        if time_cols:
            time_col = time_cols[0]
            bytes_cols = self.detected_features['bytes_columns']
            packets_cols = self.detected_features['packets_columns']

            if bytes_cols:
                df['bytes_per_second'] = df[bytes_cols[0]] / (df[time_col] + epsilon)
                time_features_added += 1

            if packets_cols:
                df['packets_per_second'] = df[packets_cols[0]] / (df[time_col] + epsilon)
                time_features_added += 1

            # دسته‌بندی مدت زمان
            df['duration_category'] = pd.cut(
                df[time_col],
                bins=[0, 1, 30, float('inf')],
                labels=['short', 'medium', 'long'],
                right=False
            )
            time_features_added += 1

            # پرچم‌های دسته زمان
            df['is_short_session'] = (df[time_col] < 1).astype(int)
            df['is_long_session'] = (df[time_col] > 30).astype(int)
            time_features_added += 2

        print(f"   ✅ {time_features_added} ویژگی زمانی ایجاد شد")
        return df

    def _apply_log_transforms(self, df):
        """اعمال تبدیل لگاریتمی برای کاهش skew - عمومی"""
        print("🔧 اعمال تبدیل لگاریتمی...")

        log_features_added = 0

        # ستون‌های عددی با skew بالا (به جز دسته‌ای و پرچم‌ها)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['category', 'is_', 'ratio', 'match', 'has_', 'flag']

        for col in numeric_cols:
            if any(pattern in col for pattern in exclude_patterns):
                continue

            # محاسبه skew و اعمال log1p اگر skew بالا باشد
            skew_val = df[col].skew()
            if abs(skew_val) > 1.0:  # آستانه skew
                df[f'log1p_{col}'] = np.log1p(df[col])
                log_features_added += 1

        print(f"   ✅ {log_features_added} تبدیل لگاریتمی اعمال شد")
        return df

    def _encode_categorical_features(self, df, is_train=True):
        """رمزگذاری ویژگی‌های دسته‌ای - عمومی"""
        print("🔤 رمزگذاری ویژگی‌های دسته‌ای...")

        # شناسایی ستون‌های دسته‌ای (غیرعددی)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        encoded_features = 0

        # ۱. ابتدا ستون هدف را encode کنیم
        if self.target_column in df.columns and df[self.target_column].dtype in ['object', 'category']:
            if is_train:
                target_encoder = LabelEncoder()
                df[self.target_column] = target_encoder.fit_transform(df[self.target_column].astype(str))
                self.encoders[self.target_column] = target_encoder
            else:
                if self.target_column in self.encoders:
                    # مدیریت مقادیر جدید در test set
                    unique_vals = set(df[self.target_column].astype(str).unique())
                    trained_vals = set(self.encoders[self.target_column].classes_)
                    new_vals = unique_vals - trained_vals

                    if new_vals:
                        print(f"   ⚠️  مقادیر جدید در ستون هدف: {new_vals}")
                        df[self.target_column] = df[self.target_column].astype(str)
                        df.loc[df[self.target_column].isin(new_vals), self.target_column] = 'unknown'

                    df[self.target_column] = self.encoders[self.target_column].transform(
                        df[self.target_column].astype(str))
                else:
                    raise Exception(f"Encoder برای ستون هدف '{self.target_column}' یافت نشد")

            encoded_features += 1
            print(f"   ✅ {self.target_column}: Label Encoding (ستون هدف)")

        # ۲. حالا سایر ستون‌های دسته‌ای را encode کنیم (بدون حذف ستون هدف)
        for col in categorical_cols:
            if col == self.target_column:
                continue

            if df[col].nunique() <= 10:  # One-Hot برای دسته‌های کم
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                encoded_features += len(dummies.columns)
                print(f"   ✅ {col}: One-Hot Encoding ({len(dummies.columns)} ویژگی)")
            else:  # Label Encoding برای دسته‌های زیاد
                if is_train:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    if col in self.encoders:
                        # مدیریت مقادیر جدید در test set
                        unique_vals = set(df[col].astype(str).unique())
                        trained_vals = set(self.encoders[col].classes_)
                        new_vals = unique_vals - trained_vals

                        if new_vals:
                            df[col] = df[col].astype(str)
                            df.loc[df[col].isin(new_vals), col] = 'unknown'

                        df[col] = self.encoders[col].transform(df[col].astype(str))
                    else:
                        # اگر encoder وجود ندارد، حذف ستون
                        df = df.drop(columns=[col])
                encoded_features += 1
                print(f"   ✅ {col}: Label Encoding")

        print(f"   ✅ در مجموع {encoded_features} ویژگی دسته‌ای رمزگذاری شد")
        return df

    def _scale_features(self, df, fit_scalers=True):
        """مقیاس‌بندی ویژگی‌های عددی - عمومی"""
        print("📊 مقیاس‌بندی ویژگی‌های عددی...")

        # شناسایی ستون‌های عددی (به جز هدف و پرچم‌ها)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # حذف ستون هدف و پرچم‌های باینری
        exclude_patterns = [self.target_column] + [col for col in numeric_cols
                                                   if any(x in col for x in ['is_', 'has_', 'match', 'flag'])]
        scale_cols = [col for col in numeric_cols if col not in exclude_patterns]

        if not scale_cols:
            return df

        print(f"   📈 مقیاس‌بندی {len(scale_cols)} ویژگی عددی")

        if fit_scalers:
            # آموزش scaler جدید
            if self.scaling_strategy == 'standard':
                scaler = StandardScaler()
            elif self.scaling_strategy == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            self.scalers['numerical'] = scaler
        else:
            # استفاده از scaler آموزش دیده
            if 'numerical' in self.scalers:
                df[scale_cols] = self.scalers['numerical'].transform(df[scale_cols])
            else:
                print("⚠️  هیچ scaler آموزش دیده‌ای یافت نشد")

        return df

    def _select_features(self, df, is_train=True, k_features=50):
        """انتخاب ویژگی‌های برتر - عمومی"""
        print("🎯 انتخاب ویژگی‌های برتر...")

        if self.target_column not in df.columns:
            return df

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # ۱. حذف ویژگی‌های با واریانس پایین
        if is_train:
            variance_selector = VarianceThreshold(threshold=0.01)
            X_selected = variance_selector.fit_transform(X)
            low_var_cols = X.columns[~variance_selector.get_support()].tolist()

            if low_var_cols:
                print(f"   🗑️  حذف {len(low_var_cols)} ویژگی با واریانس پایین")
                X = X.drop(columns=low_var_cols)

        # ۲. انتخاب k ویژگی برتر
        if is_train:
            k = min(k_features, X.shape[1])
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            if self.feature_selector:
                X_selected = self.feature_selector.transform(X)
                selected_features = X.columns[self.feature_selector.get_support()].tolist()
            else:
                selected_features = X.columns.tolist()
                X_selected = X.values

        print(f"   ✅ انتخاب {len(selected_features)} ویژگی برتر")

        # ایجاد DataFrame نهایی
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        df_selected[self.target_column] = y.values

        return df_selected

    # در کلاس SmartFeatureEngineer، متد get_feature_summary را اصلاح کنید:

    def get_feature_summary(self):
        """گزارش خلاصه ویژگی‌های ایجاد شده """
        # تبدیل به انواع استاندارد پایتون
        detected_features_serializable = {}
        for key, value in self.detected_features.items():
            detected_features_serializable[key] = [str(item) for item in value] if isinstance(value, list) else value

        return {
            'detected_features': detected_features_serializable,
            'fitted': self.fitted,
            'scaling_strategy': self.scaling_strategy,
            'encoders_count': int(len(self.encoders)),
            'scalers_count': int(len(self.scalers)),
            'has_feature_selector': self.feature_selector is not None
        }