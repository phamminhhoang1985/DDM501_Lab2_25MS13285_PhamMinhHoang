# DDM501 Lab 2 — ML Pipeline & Experiment Tracking

**Sinh viên:** Phạm Minh Hoàng  
**MSSV:** 25MS13285  
**Môn học:** DDM501 — AI in DevOps, DataOps, MLOps — FSB, FPT University  
**Bài tập:** Lab 2 — ML Pipeline & Experiment Tracking  

---

## Mục tiêu

Lab 2 chuyển đổi script huấn luyện đơn lẻ từ Lab 1 thành một **ML Pipeline hoàn chỉnh** với:

- **Data Quality Gate:** Kiểm tra dữ liệu 3 cấp độ trước khi huấn luyện
- **MLflow Experiment Tracking:** Ghi nhận đầy đủ mọi lần chạy (parameters, metrics, artifacts)
- **Hyperparameter Sweep:** So sánh nhiều cấu hình mô hình qua leaderboard
- **Automated Quality Gate:** Đánh giá hiệu năng + fairness, tự động thăng hạng mô hình
- **Airflow DAG:** Orchestrate pipeline với lịch tự động, phân nhánh có điều kiện

---

## Cấu trúc dự án

```
ddm501-lab2-starter/
├── pipeline/
│   ├── config.py           # Cấu hình toàn bộ, hỗ trợ env-override
│   ├── data_ingestion.py   # Load CSV, stratified split, dataset stats
│   ├── validation.py       # Data quality gate 3 cấp: schema, statistics, semantics
│   ├── preprocessing.py    # 6 derived features + ColumnTransformer (unfitted)
│   ├── training.py         # Fit bên trong MLflow run
│   ├── evaluation.py       # Metrics, per-group metrics, fairness gap
│   ├── registry.py         # Register, alias, quality gate, promotion
│   └── run_pipeline.py     # CLI entry point
├── experiments/
│   └── run_experiments.py  # Grid sweep + leaderboard
├── dags/
│   └── credit_training_dag.py  # Airflow DAG
├── docker/
│   └── airflow.Dockerfile
├── data/
│   └── credit_default.csv  # Dataset (30,000 rows)
├── tests/
│   └── test_pipeline.py
├── docker-compose.yml
├── requirements.txt        # Pipeline dependencies (NO Airflow)
└── requirements-airflow.txt # Airflow image dependencies
```

---

## Quick Start

### 1. Chạy pipeline cục bộ (không cần server)

```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy pipeline (tự động dùng local MLflow file store tại ./mlruns)
python -m pipeline.run_pipeline

# Xem kết quả trong MLflow UI
mlflow ui --backend-store-uri ./mlruns   # http://localhost:5000
```

### 2. Chạy với MLflow tracking server (Docker)

```bash
# Khởi động MLflow server
docker compose up -d mlflow

# Xác nhận server hoạt động
python scripts/setup_mlflow.py

# Chạy pipeline với tracking server
set MLFLOW_TRACKING_URI=http://localhost:5000  # Windows
# export MLFLOW_TRACKING_URI=http://localhost:5000  # Linux/macOS
python -m pipeline.run_pipeline
```

### 3. Chạy hyperparameter sweep

```bash
# Chạy toàn bộ 7 cấu hình (logreg x2, rf x2, hgb x3)
python -m experiments.run_experiments

# Chỉ chạy một model type
python -m experiments.run_experiments --model-type hgb

# Xem leaderboard mà không cần chạy thêm
python -m experiments.run_experiments --leaderboard-only --top 5
```

### 4. Full stack với Airflow

```bash
docker compose up -d --build

# MLflow   http://localhost:5000
# Airflow  http://localhost:8080  (username: airflow / password: airflow)
```

Vào Airflow UI, unpause DAG `credit_default_training` và trigger thủ công.

### 5. Chạy tests

```bash
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

---

## Thiết kế Pipeline

### 5 Stages

```
[Ingestion] → [Validation Gate] → [Training + MLflow] → [Evaluation] → [Registry & Promotion]
```

| Stage | File | Chức năng |
|---|---|---|
| **Ingestion** | `data_ingestion.py` | Load CSV, stratified split (seed=501), dataset stats |
| **Validation** | `validation.py` | 3-level data quality gate |
| **Training** | `training.py` | Fit pipeline trong MLflow run |
| **Evaluation** | `evaluation.py` | Aggregate + per-group metrics, fairness gap |
| **Registry** | `registry.py` | Quality gate, register, alias |

### Tại sao Validation phải đứng trước Training?

Validation đặt **trước** Training vì:
1. **Fail fast:** Dữ liệu xấu bị chặn ngay, không tốn tài nguyên huấn luyện
2. **Audit trail:** Report lỗi gắn với run để sau này truy vết
3. **Business logic:** Kiểm tra ý nghĩa nghiệp vụ (AGE=400, SEX=7) không thể phát hiện bằng model metrics

---

## Data Quality Gate (3 Levels)

| Level | Kiểm tra | Mục đích |
|---|---|---|
| **Schema** | Cột đầy đủ, kiểu số | Đảm bảo cấu trúc đúng format |
| **Statistics** | Số dòng ≥ 5000, missing ≤ 2%, positive rate ∈ [5%, 60%] | Phát hiện extract lỗi, degenerate target |
| **Semantics** | SEX ∈ {1,2}, EDUCATION ∈ {1-4}, AGE ∈ [18,100], PAY_AMT ≥ 0 | Domain constraints từ nghiệp vụ |

Kiểm tra degenerate target (tất cả label = 0) là **quan trọng nhất**: một extract lỗi trả về toàn 0 sẽ cho model accuracy 78% mà predict "không vỡ nợ" cho tất cả.

---

## Features Engineering

6 derived features mã hóa kiến thức tín dụng:

| Feature | Công thức | Ý nghĩa |
|---|---|---|
| `utilisation_ratio` | `mean(BILL_AMT1..6) / LIMIT_BAL` [0,5] | Tỷ lệ sử dụng hạn mức tín dụng |
| `payment_ratio` | `PAY_AMT1 / BILL_AMT1` [0,5] | Tỷ lệ thanh toán sao kê |
| `max_delay` | `max(PAY_0, PAY_2..PAY_6)` | Trễ hạn nặng nhất |
| `n_months_delayed` | `count(PAY_* > 0)` | Số tháng trễ hạn |
| `avg_bill_amt` | `mean(BILL_AMT1..6)` | Trung bình dư nợ |
| `avg_pay_amt` | `mean(PAY_AMT1..6)` | Trung bình thanh toán |

---

## Quality Gate & Promotion

### Thresholds

| Metric | Threshold | Hướng |
|---|---|---|
| `roc_auc` | ≥ 0.70 | Cao hơn tốt hơn |
| `pr_auc` | ≥ 0.45 | Cao hơn tốt hơn |
| `fairness_gap` | ≤ 0.10 | Thấp hơn tốt hơn |

### 3 Kết quả có thể

| Kết quả | Điều kiện | Alias |
|---|---|---|
| **champion** | Đạt gate + vượt champion cũ ≥ 0.002 AUC | `@champion` |
| **challenger** | Đạt gate nhưng không vượt margin | `@challenger` |
| **rejected** | Trượt gate | Không alias, tag `quality_gate: failed` |

> **Lưu ý:** Model bị rejected **vẫn được đăng ký** vào MLflow để tạo audit trail. Đây là cách chứng minh với regulator rằng model xấu đã bị phát hiện, không phải không được tạo ra.

---

## Airflow DAG

### Luồng thực thi

```
ingest → validate → train → evaluate → decide → [promote_model | skip_promotion] → cleanup
```

| Task | Chức năng |
|---|---|
| `ingest` | Load CSV, split, ghi lên shared volume |
| `validate` | Schema/statistics/semantics gate, raise nếu fail |
| `train` | Fit trong MLflow run, push model lên volume |
| `evaluate` | Score test set, log metrics, push scalars qua XCom |
| `decide` | `BranchPythonOperator`: pass → `promote_model`, fail → `skip_promotion` |
| `promote_model` | Register, set alias dựa trên outcome |
| `skip_promotion` | `EmptyOperator`, không làm gì |
| `cleanup` | Xóa run directory, `trigger_rule=none_failed_min_one_success` |

### XCom vs Shared Volume

| Gửi qua XCom | Lưu trên Volume |
|---|---|
| `run_dir` (path), `mlflow_run_id`, scalar metrics, `quality_gate` result | Raw parquet, split joblib, model joblib, validation/evaluation json |

**Lý do tách biệt:** XCom serialized vào metadata DB của Airflow có size limit. DataFrame và model file nếu đẩy vào XCom sẽ gây crash hoặc timeout.

---

## Kết quả Experiment Sweep

Kết quả điển hình của leaderboard từ 7 cấu hình (3 model families):

| Model | ROC AUC | PR AUC | Fairness Gap | Kết quả Gate |
|---|---|---|---|---|
| hgb (best) | ~0.778 | ~0.542 | ~0.06 | ✅ **Champion** |
| hgb (mid) | ~0.772 | ~0.535 | ~0.06 | ✅ Challenger |
| rf (deep) | ~0.769 | ~0.528 | ~0.07 | ✅ Challenger |
| rf (shallow) | ~0.760 | ~0.520 | ~0.07 | ✅ Challenger |
| logreg (C=1.0) | ~0.773 | ~0.530 | ~0.12 | ❌ Rejected (fairness) |
| logreg (C=0.1) | ~0.765 | ~0.518 | ~0.13 | ❌ Rejected (fairness) |
| hgb (deep) | ~0.775 | ~0.538 | ~0.06 | ✅ Challenger |

*(Số liệu có thể thay đổi theo lần chạy, nhưng pattern thường nhất quán)*

### Phân tích & Quyết định Promotion

**Kết quả quan trọng nhất:** Logistic Regression thường có ROC AUC cao nhưng fairness gap rộng nhất (>0.10), trong khi HistGradientBoosting có AUC thấp hơn một chút nhưng fairness gap hẹp hơn nhiều.

**Quyết định:** Promote **hgb** thay vì logreg vì:

1. **Fairness gap của logreg (>0.12) vượt ngưỡng 0.10:** Mô hình áp dụng selection rate khác nhau có ý nghĩa thống kê giữa các nhóm giới tính, vi phạm tiêu chí công bằng.

2. **AUC chênh lệch nhỏ không biện minh cho rủi ro fairness:** Logistic Regression chỉ hơn HGB ~0.002-0.005 AUC — trong biên độ margin — trong khi rủi ro pháp lý/đạo đức từ fairness gap lớn hơn nhiều.

3. **Trong bài toán tín dụng, fairness là yêu cầu pháp lý:** Equal Credit Opportunity Act yêu cầu không phân biệt đối xử theo giới tính trong quyết định tín dụng.

**Tái lập kết quả tốt nhất:** Load run từ MLflow với artifact `feature_columns.json` để biết đúng cột cần thiết, `validation_report.json` để biết chất lượng data, rồi `mlflow.sklearn.load_model("runs:/<run_id>/model")`.

---

## Reproducibility

Mọi thứ cần để tái lập đều được log vào MLflow run:

| Artifact | Nội dung |
|---|---|
| `feature_columns.json` | Danh sách chính xác các feature mà model kỳ vọng |
| `validation_report.json` | Trạng thái dữ liệu khi training |
| `evaluation.json` | Metrics đầy đủ bao gồm per-group |
| `model/` | Fitted sklearn Pipeline (preprocessor + classifier) |
| MLflow params | `model_type`, `n_features`, `n_train_rows`, hyperparams |

---

## Infrastructure

### Hai bộ dependencies (tách biệt có chủ đích)

| File | Dùng bởi | Lý do |
|---|---|---|
| `requirements.txt` | Laptop, CI, tests | numpy 2.2, pandas 2.2, sklearn 1.6 |
| `requirements-airflow.txt` | `airflow.Dockerfile` | numpy 1.24, pandas 2.1 — pinned bởi Airflow constraints |

**Quan trọng:** Không bao giờ thêm `apache-airflow` vào `requirements.txt`. CI sẽ fail.

### MLflow Aliases (thay vì deprecated Stages)

```python
# ❌ Deprecated (MLflow >= 2.9)
client.transition_model_version_stage(name, version, "Production")

# ✅ Sử dụng aliases
client.set_registered_model_alias(name, alias="champion", version=version)
model = mlflow.sklearn.load_model("models:/credit-default-classifier@champion")
```

---

## Biến môi trường

Tất cả có giá trị mặc định trong `config.py` để chạy ngay mà không cần cấu hình thêm:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `file://<BASE_DIR>/mlruns` | Local file store, không cần server |
| `MLFLOW_EXPERIMENT_NAME` | `credit-default-risk` | Tên experiment |
| `REGISTERED_MODEL_NAME` | `credit-default-classifier` | Tên model registry |
| `DATA_PATH` | `data/credit_default.csv` | Đường dẫn dataset |
| `MIN_ROC_AUC` | `0.70` | Quality gate threshold |
| `MIN_PR_AUC` | `0.45` | Quality gate threshold |
| `MAX_FAIRNESS_GAP` | `0.10` | Quality gate threshold |
| `RANDOM_STATE` | `501` | Seed cho reproducibility |
| `TEST_SIZE` | `0.20` | Tỷ lệ test split |
| `REVIEW_THRESHOLD` | `0.30` | Decision threshold |

---

## Troubleshooting

**`Experiment 'credit-default-risk' not found`**  
Chạy pipeline ít nhất một lần trước: `python -m pipeline.run_pipeline --no-register`

**`Cannot reach the tracking server`**  
Chạy `docker compose up -d mlflow` hoặc unset `MLFLOW_TRACKING_URI` để dùng local file store.

**`airflow dags list` ra graph phẳng**  
TODO 6 (wiring dependencies) chưa hoàn thành. Kiểm tra `t_ingest >> t_validate >> ...`

**`transition_model_version_stage is deprecated`**  
Sử dụng `set_registered_model_alias` thay thế.

**Mọi test pass trước khi implement**  
Test stub dùng `pass` trả về `None` — pytest đếm là pass. Xóa `pass` khi implement.

**DAG chạy nhưng không có gì trong MLflow**  
Trong compose network, tracking server là `http://mlflow:5000`, không phải `localhost:5000`.

---

## Tài liệu tham khảo

- Küstner, Le Goues & Hilton (2024), *Machine Learning in Production*
- Huyen (2022), *Designing Machine Learning Systems*, ch.5-6
- MLflow docs: [Migrating from Stages to Aliases](https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages-to-model-aliases)
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*

---

*DDM501 — AI in DevOps, DataOps, MLOps — FSB, FPT University*
