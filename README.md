# ML Thermal DAG Scheduler

زمان‌بندی DAG بر اساس مقالهٔ Sub-Graph Division و تولید دادهٔ حرارتی مصنوعی برای همان کارها.

## نصب و اجرا

Python 3.10 یا بالاتر:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_demo.py
```

خروجی مثال مقاله در `outputs/paper_demo/` ذخیره می‌شود:

| سنجه | مقدار |
| --- | --- |
| بار کل | 27 |
| طول مسیر بحرانی | 14 |
| زمان پایان شبیه‌سازی | 15 |
| مقدار الگوریتم ۳ | 16.5 |
| کران Graham | 20.5 |

الگوریتم ۳ فعلاً آزمایشی است: برای بعضی گراف‌ها کمتر از زمان شبیه‌سازی نتیجه
می‌دهد. `algorithm_3_status` و `sanity_issues` این موارد را مشخص می‌کنند.
برای مقایسه با deadline از `model_safe_bound`، یعنی کران Graham، استفاده می‌شود.
نمونهٔ ناسازگاری:

```sh
python run_demo.py --input examples/algorithm_3_counterexample.json --cores 2
```

## تولید DAG

```sh
python dag_scheduler.py --count 10 --seed 2026 --parallelism 8 \
  --workload 500 --out outputs/generated
python run_demo.py --input outputs/generated/dag_0000.json --cores 4
```

عمق پیش‌فرض ۵ تا ۸ و عرض لایه‌های میانی ۲ تا `p` است. سیاست یال‌ها از
[dag-gen-rnd v0.1](https://github.com/automaticdai/dag-gen-rnd/tree/cd8df21a48047610234aaf990141ea128cdfb8bc)
گرفته شده؛ مجوز در `third_party/dag-gen-rnd-LICENSE` است. احتمال یال ۰٫۵،
پیش‌فرض مرجع است و مقدار مشخص‌شده در مقاله نیست. WCETها با UUniFast-discard
به‌صورت پیوسته تولید می‌شوند؛ بذر، بار کل و تنظیمات در JSON ثبت می‌شوند.
`--max-wcet` سقف اختیاری است. با `--period`، دوره و deadline برابر ثبت می‌شوند؛
بدون آن، هر دو نامشخص‌اند. واحد پیش‌فرض زمان انتزاعی است.

## دادهٔ حرارتی

```sh
python run_demo.py --thermal --seconds-per-unit 0.01 \
  --thermal-dt 0.25 --coupling 0.15
```

در این دستور هر واحد زمان را به‌صورت فرضی ۰٫۰۱ ثانیه می‌گیریم. مدل RC در مرز
شروع و پایان کارها حل می‌شود؛ دمای هسته بین کارها حفظ می‌شود. `coupling`
رسانایی حرارتی بین همسایه‌های یک grid مصنوعی است؛ صفر یعنی هسته‌های مستقل.

خروجی شامل رد توان/دما، نگاشت توان هر کار، دادهٔ یادگیری یک‌گام‌جلو، پارامترها و
نمودار است. ورودی‌های یادگیری فقط از زمان فعلی‌اند و هدف، دمای نمونهٔ بعدی است.
همسایه‌های ناموجود mask دارند و فاصلهٔ زمانی هدف در `horizon_seconds` ثبت می‌شود.

پارامترهای حرارتی مصنوعی و کالیبره‌نشده‌اند. ANN+LMS و کنترل حرارتی هنوز پیاده
نشده‌اند؛ ستون‌های throttling در دادهٔ فعلی صفرند. فایل `ann_lms_predictor.py`
فعلاً مولد قدیمی است و در مسیر اجرای این بخش استفاده نمی‌شود.

## اجرای گروهی

```sh
python run_experiments.py --count 1000 --cores 1 2 4 --parallelism 8 \
  --out outputs/experiments
```

هر تعداد هسته، همان ورودی را برای هر بذر می‌گیرد. موارد نقض و خطا نیز در نتایج
حفظ می‌شوند. نبودن نقض در شبیه‌سازی اثبات کران نیست. مقایسهٔ He2019/Zhao2020
انجام نشده و تعداد مسیرهای محلی هنوز سقف ۱۰هزار مسیر برای هر بلوک دارد.

منبع اصلی: Zhang, Zhao & Huang,
[DAG Scheduling and Analysis in Multicore Real-Time Systems via Sub-Graph Division](https://doi.org/10.1109/TCAD.2026.3682647).
