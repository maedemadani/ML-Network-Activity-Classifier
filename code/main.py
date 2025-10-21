import os
import sys
import argparse
from pathlib import Path


def setup_environment():
    """تنظیم محیط و مسیرها"""
    # اضافه کردن مسیر src به sys.path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

    # ایجاد پوشه‌های ضروری
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("✅ محیط اجرا تنظیم شد")

def run_phase1():
    """اجرای فاز ۱: پاکسازی داده‌ها"""
    print("\n" + "=" * 60)
    print("فاز ۱: پاکسازی و آماده‌سازی داده‌ها")
    print("=" * 60)

    try:
        from src.phase1_data_preparation import main as phase1_main

        # گرفتن ورودی از کاربر
        print("\n📥 تنظیمات فاز ۱:")
        input_file = input("مسیر فایل ورودی [پیش‌فرض: network_logs.csv]: ").strip()
        if not input_file:
            input_file = "network_logs.csv"

        output_dir = input("پوشه خروجی [پیش‌فرض: data/cleanedData]: ").strip()
        if not output_dir:
            output_dir = "data/cleanedData"

        print("روش مدیریت outliers:")
        print("1. remove - حذف outlierها")
        print("2. mark - علامت‌گذاری outlierها")
        print("3. clip - محدود کردن outlierها")
        print("4. ignore - نادیده گرفتن outlierها")
        outlier_choice = input("انتخاب شما [پیش‌فرض: 1]: ").strip()

        outlier_methods = {'1': 'remove', '2': 'mark', '3': 'clip', '4': 'ignore'}
        outlier_method = outlier_methods.get(outlier_choice, 'remove')

        # allow_negative = input("ستون‌های مجاز برای مقادیر منفی (جدا با space) [پیش‌فرض: هیچ‌کدام]: ").strip()
        # allow_negative_cols = allow_negative.split() if allow_negative else []

        # اجرای فاز ۱
        print(f"\n🚀 اجرای فاز ۱ با تنظیمات:")
        print(f"   فایل ورودی: {input_file}")
        print(f"   پوشه خروجی: {output_dir}")
        print(f"   روش outliers: {outlier_method}")
        # print(f"   ستون‌های مجاز منفی: {allow_negative_cols}")

        result = phase1_main(
            data_path=input_file,
            output_dir=output_dir,
            outlier_method=outlier_method,
            # allow_negative_cols=allow_negative_cols
        )

        if result and result.get('success'):
            print("✅ فاز ۱ با موفقیت completed شد!")
            return True
        else:
            print("❌ فاز ۱ با خطا مواجه شد")
            return False

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۱: {e}")
        return False

def run_phase2():
    """اجرای فاز ۲: مهندسی ویژگی"""
    print("\n" + "=" * 60)
    print("فاز ۲: مهندسی ویژگی و مقیاس‌بندی")
    print("=" * 60)

    try:
        from src.phase2_feature_engineering import main as phase2_main

        # گرفتن ورودی از کاربر
        print("\n📥 تنظیمات فاز ۲:")
        input_file = input("فایل ورودی [پیش‌فرض: data/cleanedData/network_logs_cleaned.csv]: ").strip()
        if not input_file:
            input_file = "data/cleanedData/network_logs_cleaned.csv"

        metadata_file = input("فایل متادیتا [پیش‌فرض: data/columns_metadata.json]: ").strip()
        if not metadata_file:
            metadata_file = "data/columns_metadata.json"

        print("استراتژی مقیاس‌بندی:")
        print("1. standard - StandardScaler")
        print("2. robust - RobustScaler")
        scaling_choice = input("انتخاب شما [پیش‌فرض: 1]: ").strip()
        scaling_methods = {'1': 'standard', '2': 'robust'}
        scaling_strategy = scaling_methods.get(scaling_choice, 'standard')

        # اجرای فاز ۲
        print(f"\n🚀 اجرای فاز ۲ با تنظیمات:")
        print(f"   فایل ورودی: {input_file}")
        print(f"   فایل متادیتا: {metadata_file}")
        print(f"   استراتژی مقیاس‌بندی: {scaling_strategy}")

        result = phase2_main(
            input_file=input_file,
            metadata_file=metadata_file,
            scaling_strategy=scaling_strategy
        )

        if result is not None:
            print("✅ فاز ۲ با موفقیت completed شد!")
            return True
        else:
            print("❌ فاز ۲ با خطا مواجه شد")
            return False

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۲: {e}")
        return False

def run_phase3():
    """اجرای فاز ۳: مدیریت عدم تعادل"""
    print("\n" + "=" * 60)
    print("فاز ۳: مدیریت عدم تعادل کلاس‌ها")
    print("=" * 60)

    try:
        from src.phase3_balancing import main as phase3_main

        # گرفتن ورودی از کاربر
        print("\n📥 تنظیمات فاز ۳:")
        input_file = input("فایل ورودی [پیش‌فرض: data/engineeredData/engineered_dataset.csv]: ").strip()
        if not input_file:
            input_file = "data/engineeredData/engineered_dataset.csv"

        output_dir = input("پوشه خروجی [پیش‌فرض: data/balancedData]: ").strip()
        if not output_dir:
            output_dir = "data/balancedData"

        # اجرای فاز ۳
        print(f"\n🚀 اجرای فاز ۳ با تنظیمات:")
        print(f"   فایل ورودی: {input_file}")
        print(f"   پوشه خروجی: {output_dir}")

        result = phase3_main(
            input_file=input_file,
            output_dir=output_dir
        )

        if result:
            print("✅ فاز ۳ با موفقیت completed شد!")
            return True
        else:
            print("❌ فاز ۳ با خطا مواجه شد")
            return False

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۳: {e}")
        return False

def run_phase4():
    """اجرای فاز ۴: مدل‌سازی"""
    print("\n" + "=" * 60)
    print("فاز ۴: مدل‌سازی و ارزیابی")
    print("=" * 60)

    try:
        from src.phase4_model_training import main as phase4_main

        print("🚀 اجرای فاز ۴...")
        result = phase4_main()

        if result is not None:
            print("✅ فاز ۴ با موفقیت completed شد!")
            return True
        else:
            print("❌ فاز ۴ با خطا مواجه شد")
            return False

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۴: {e}")
        return False


def run_phase5():
    """اجرای فاز ۵: تحلیل نهایی"""
    print("\n" + "=" * 60)
    print("فاز ۵: تحلیل نهایی و تحویل")
    print("=" * 60)

    try:
        from src.phase5_final_analysis import main as phase5_main

        print("🚀 اجرای فاز ۵...")
        result = phase5_main()

        if result is not None:
            print("✅ فاز ۵ با موفقیت completed شد!")
            return True
        else:
            print("❌ فاز ۵ با خطا مواجه شد")
            return False

    except Exception as e:
        print(f"❌ خطا در اجرای فاز ۵: {e}")
        return False


def wait_for_enter(phase_name):
    """انتظار برای فشردن اینتر توسط کاربر"""
    print(f"\n⏎ برای ادامه به فاز {phase_name}، کلید Enter را بفشارید...")
    input()


def print_banner():
    """چاپ بنر پروژه"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   پروژه طبقه‌بندی فعالیت‌های شبکه              ║
    ║             Network Activity Classification                  ║
    ╚══════════════════════════════════════════════════════════════╝

    اجرای متوالی تمام فازهای پروژه
    """
    print(banner)


def main():
    """تابع اصلی"""
    print_banner()
    setup_environment()

    # تعریف فازها
    phases = [
        (1, "پاکسازی داده‌ها", run_phase1),
        (2, "مهندسی ویژگی", run_phase2),
        (3, "مدیریت عدم تعادل", run_phase3),
        (4, "مدل‌سازی", run_phase4),
        (5, "تحلیل نهایی", run_phase5)
    ]

    # اجرای فازها
    successful_phases = []

    for phase_num, phase_name, phase_func in phases:
        try:
            # اجرای فاز
            success = phase_func()

            if success:
                successful_phases.append(phase_num)

            # اگر آخرین فاز نیست، منتظر اینتر بمان
            if phase_num < len(phases):
                next_phase_num = phase_num + 1
                next_phase_name = phases[next_phase_num - 1][1]
                wait_for_enter(next_phase_name)

        except KeyboardInterrupt:
            print(f"\n⏹️  اجرای پروژه توسط کاربر متوقف شد")
            break
        except Exception as e:
            print(f"❌ خطای غیرمنتظره در فاز {phase_num}: {e}")
            break

    # گزارش نهایی
    print("\n" + "=" * 60)
    print("گزارش نهایی اجرای پروژه")
    print("=" * 60)

    if len(successful_phases) == len(phases):
        print("🎉 تمام فازهای پروژه با موفقیت به پایان رسید!")
    else:
        print(f"📊 {len(successful_phases)} از {len(phases)} فاز با موفقیت اجرا شدند")
        print(f"✅ فازهای موفق: {successful_phases}")

        failed_phases = [i for i in range(1, len(phases) + 1) if i not in successful_phases]
        if failed_phases:
            print(f"❌ فازهای ناموفق: {failed_phases}")

    print("\n👋 خروج از برنامه")


if __name__ == "__main__":
    # پارسر برای حالت غیرتعاملی
    parser = argparse.ArgumentParser(description='اجرای متوالی تمام فازهای پروژه طبقه‌بندی شبکه')
    parser.add_argument('--non-interactive', action='store_true',
                        help='اجرای غیرتعاملی')

    args = parser.parse_args()

    if args.non_interactive:
        print("🔧 اجرای غیرتعاملی - استفاده از مقادیر پیش‌فرض")
        # در این حالت می‌توانید مستقیماً توابع را با مقادیر پیش‌فرض فراخوانی کنید
        # اما برای سادگی، همان حالت تعاملی را اجرا می‌کنیم
        # با این تفاوت که ورودی‌ها به صورت خودکار پر شوند

    main()