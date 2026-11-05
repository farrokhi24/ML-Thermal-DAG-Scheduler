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
برای مقایسه با deadline از `certified_response_bound` استفاده می‌شود: بار کل
برای یک هسته، طول مسیر بحرانی در صورت کفایت تعداد هسته‌ها، و Graham در حالت
عمومی. `model_safe_bound` برای سازگاری همچنان مقدار Graham را دارد.
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

پارامترهای حرارتی مصنوعی و کالیبره‌نشده‌اند. در این اجرای بدون کنترل، ستون‌های
throttling صفرند؛ برای دادهٔ دارای تغییر فعالیت از مسیر زیر استفاده کنید.

## اجرای کامل با ANN+LMS و کنترل دما

```sh
python run_system.py --seconds-per-unit 0.05 --dt 1 --cores 2 --limit 55
```

این دستور روی گراف مقاله اجرا می‌شود. برای یک ورودی بزرگ‌تر:

```sh
python dag_scheduler.py --count 1 --seed 2026 --workload 500 --out outputs/generated
python run_system.py --input outputs/generated/dag_0000.json \
  --seconds-per-unit 0.05 --out outputs/system
```

آموزش اولیه روی DAGهای مصنوعی مستقل با نسبت فعالیت متغیر انجام می‌شود. ارزیابی
پیش‌بینی روی یک DAG مستقل و بدون کنترل، ANN ثابت، ANN+LMS و persistence را
مقایسه می‌کند. اجرای پایه و کنترل‌شده، گراف، توان هر کار و شرایط اولیهٔ یکسان دارند.

پیش‌بین هر هسته یک نورون با ReLU و ۱۸ ورودی است: دمای خود و هشت همسایه،
به‌علاوهٔ نسبت throttling همان ۹ هسته. وزن‌ها ابتدا با کمترین مربعات منظم‌شده
آموزش می‌بینند و پس از رسیدن دمای هدف با LMS اصلاح می‌شوند. مقیاس دما ۱۰۰ و
مقدار پیش‌فرض μ برابر فاصلهٔ حسگری بر حسب ثانیه، روی ورودی‌های مقیاس‌شده، است.
هدف آینده پیش از زمان رسیدنش وارد اصلاح وزن نمی‌شود.

سیاست `predictive` بیشترین فعالیت قابل قبول از مجموعهٔ {۱،۰٫۷۵،۰٫۵،۰٫۲۵،۰}
را با پیش‌بین انتخاب می‌کند. رسیدن دمای اندازه‌گیری‌شده به حد، cooldown سراسری
با hysteresis را فعال می‌کند. اگر پیش‌بینی نامناسب همهٔ کارها را متوقف کند، در
خارج از cooldown حداقل یک هسته با فعالیت ۰٫۲۵ پیش می‌رود. `--policy reactive`
فقط کنترل بازخوردی و `--policy none` اجرای بدون کنترل است.

کاهش فعالیت هم توان دینامیک و هم سرعت پیشرفت کار را کم می‌کند؛ کار روی همان
هسته باقی می‌ماند. DVFS و مهاجرت مدل نشده‌اند. سیاست انتخاب نسبت، cooldown و
قاعدهٔ پیشرفت، فرض‌های پیاده‌سازی ما هستند؛ مقالهٔ چهارصفحه‌ای فقط پیش‌بین
تک‌نورونی/LMS و استفاده در PDTM را مشخص می‌کند.

داده‌ها در `outputs/system/` شامل مدل، زمان‌بندی، تصمیم‌های کنترل، دما،
پیش‌بینی‌ها و سنجه‌های خطا/انرژی/زمان پایان‌اند. برای کنترل، کران محافظه‌کارانهٔ
جداگانه بر اساس مدل RC، cooldown و حداقل پیشرفت محاسبه می‌شود؛ کران Graham
و مقدار الگوریتم ۳ به اجرای بدون کنترل مربوط‌اند. حد دما با حسگری گسسته، تضمین
سختِ پیوسته نیست و موارد عبور از آن در سنجه‌ها ثبت می‌شوند.

ارزیابی یک فایل دادهٔ یادگیری موجود با جداسازی زمانی:

```sh
python ann_lms_predictor.py --learning outputs/system/baseline/thermal_trace_learning.csv \
  --cores 2 --out outputs/predictor
```

## اجرای گروهی

```sh
python run_experiments.py --count 1000 --cores 1 2 4 --parallelism 8 \
  --out outputs/experiments
```

هر تعداد هسته، همان ورودی را برای هر بذر می‌گیرد. موارد نقض و خطا نیز در نتایج
حفظ می‌شوند. نبودن نقض در شبیه‌سازی اثبات کران نیست. مقایسهٔ He2019/Zhao2020
انجام نشده است. تخصیص اولویت با برنامه‌ریزی پویا انجام می‌شود و به فهرست‌کردن
تمام مسیرها نیاز ندارد. فقط تحلیل آزمایشی الگوریتم ۳ سقف ۱۰هزار مسیر برای هر
بلوک دارد؛ با عبور از آن، وضعیت `not_evaluated_path_limit` ثبت می‌شود و اجرای
زمان‌بند و محاسبهٔ کران معتبر ادامه می‌یابد.

منبع اصلی: Zhang, Zhao & Huang,
[DAG Scheduling and Analysis in Multicore Real-Time Systems via Sub-Graph Division](https://doi.org/10.1109/TCAD.2026.3682647).

پیش‌بین: Chen & Liao,
[Adaptive Machine Learning-based Temperature Prediction Scheme for Thermal-aware NoC System](https://doi.org/10.1109/ISCAS45731.2020.9180475)، بخش III، روابط ۳ تا ۵.
