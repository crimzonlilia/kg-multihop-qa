# Run Commands Guide

Hướng dẫn nhanh để chạy các mode `triples_3`, `triples_300`, `triples_500`, `triples_full` và đổi embedding model.

---

## 0) Form chung

```powershell
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe <script>.py [--option value] [--flag]
```

> Nếu đang đứng ở root project thì có thể thay bằng `python ...`.

---

## 1) `rebuild_graph.py`

### Form

```powershell
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe rebuild_graph.py [--samples N] [--cache NAME] [--graph-only] [--progress] [--reset]
```

### Options

- `--samples N`  
  Giới hạn số sample, ví dụ: `3`, `300`, `500`
- `--cache NAME`  
  Tên cache sẽ dùng/lưu, ví dụ: `3`, `300`, `500`, `full`
- `--graph-only`  
  Không extract lại, chỉ build graph từ cache có sẵn
- `--force-extract`  
  Bỏ qua cache hiện có và extract lại từ đầu
- `--progress`  
  Xem tiến độ resume hiện tại
- `--reset`  
  Xóa progress cũ để chạy lại từ đầu

### Ví dụ

#### Test siêu nhanh với 3 samples
```powershell
python rebuild_graph.py --samples 3 --cache 3
```

#### Build cache 300
```powershell
python rebuild_graph.py --samples 300 --cache 300
```

#### Build cache 500
```powershell
python rebuild_graph.py --samples 500 --cache 500
```

> Nếu `triples_500.json` đã tồn tại thì `python rebuild_graph.py --cache 500` giờ sẽ tự **reuse cache** và không extract lại. Muốn ép chạy lại thì thêm `--force-extract`.

#### Build full dataset
```powershell
python rebuild_graph.py --cache full
```

#### Chỉ build graph từ cache có sẵn
```powershell
python rebuild_graph.py --graph-only --cache 500
```

#### Xem resume progress
```powershell
python rebuild_graph.py --progress
```

#### Xóa progress cũ
```powershell
python rebuild_graph.py --reset
```

---

## 2) `test_full_pipeline.py`

### Form

```powershell
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe test_full_pipeline.py [--samples N] [--queries N] [--cache NAME] [--top-k K] [--embed-model MODEL]
```

### Options

- `--samples N`  
  Số sample / QA pairs load vào
- `--queries N`  
  Số query dùng để evaluate
- `--cache NAME`  
  Dùng hoặc lưu cache theo tên: `3`, `300`, `500`, `full`
- `--top-k K`  
  Retrieve top-k passages
- `--embed-model MODEL`  
  Đổi embedding model

### Embedding models hỗ trợ

- `minilm`
- `bge-small`
- `bge-base`
- `e5-small`
- `e5-base`

### Ví dụ

#### Smoke test với cache `3`
```powershell
python test_full_pipeline.py --samples 3 --queries 3 --cache 3
```

#### Chạy cache `500` với `bge-small`
```powershell
python test_full_pipeline.py --samples 500 --queries 500 --cache 500 --embed-model bge-small
```

#### Chạy cache `500` với `e5-base`
```powershell
python test_full_pipeline.py --cache 500 --embed-model e5-base
```

#### Chạy full với `top-k=20`
```powershell
python test_full_pipeline.py --cache full --top-k 20 --embed-model minilm
```

---

## 3) `check_triples.py`

### Form

```powershell
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe check_triples.py [--cache NAME] [--path FILE]
```

### Options

- `--cache NAME`  
  Chọn cache theo tên (`3`, `300`, `500`, `full`)
- `--path FILE`  
  Chỉ định file JSON cụ thể

### Ví dụ

#### Xem cache `3`
```powershell
python check_triples.py --cache 3
```

#### Xem cache `500`
```powershell
python check_triples.py --cache 500
```

#### Xem cache `full`
```powershell
python check_triples.py --cache full
```

#### Xem trực tiếp file cụ thể
```powershell
python check_triples.py --path data/cache/triples_500.json
```

---

## 4) `src/evaluate_passages.py`

Script này đang chọn cache bằng biến môi trường `TRIPLES_CACHE_NAME`.

### Form

```powershell
$env:TRIPLES_CACHE_NAME='500'
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe src/evaluate_passages.py
```

### Ví dụ

#### Chạy với cache `3`
```powershell
$env:TRIPLES_CACHE_NAME='3'
python src/evaluate_passages.py
```

#### Chạy với cache `500`
```powershell
$env:TRIPLES_CACHE_NAME='500'
python src/evaluate_passages.py
```

#### Chạy với cache `full`
```powershell
$env:TRIPLES_CACHE_NAME='full'
python src/evaluate_passages.py
```

---

## 5) `src/graph/inspect_graph.py`

Script này dùng để soi file graph `.pkl` đã lưu, rất hợp để debug nhanh mà không cần viết lệnh `python -c ...` dài dòng.

### Form

```powershell
e:/Gitcode/kg-multihop-qa/.venv/Scripts/python.exe src/graph/inspect_graph.py [--graph NAME] [--path FILE] [--top N] [--sample-edges N] [--json]
```

### Options

- `--graph NAME`  
  Chọn graph theo tên: `3`, `500`, `full`
- `--path FILE`  
  Chỉ định file graph cụ thể, ví dụ `data/processed/kg_500.pkl`
- `--top N`  
  Hiện top relation / node theo degree
- `--sample-edges N`  
  Hiện một số edge mẫu
- `--json`  
  In ra JSON thay vì dạng text dễ đọc

### Ví dụ

#### Soi graph `500`
```powershell
python src/graph/inspect_graph.py --graph 500
```

#### Soi graph `full`
```powershell
python src/graph/inspect_graph.py --graph full --top 20 --sample-edges 20
```

#### Xuất JSON stats
```powershell
python src/graph/inspect_graph.py --graph 500 --json
```

---

## 6) Lệnh nhanh nên nhớ

### Test siêu nhanh
```powershell
python test_full_pipeline.py --samples 3 --queries 3 --cache 3
```

### Test cache 500 + đổi embedding
```powershell
python test_full_pipeline.py --cache 500 --embed-model bge-small
```

### Build full
```powershell
python rebuild_graph.py --cache full
```

### Chỉ xem cache
```powershell
python check_triples.py --cache 500
```

---

## 7) Gợi ý dùng thực tế

- Muốn **check pipeline có vỡ không** → dùng `--samples 3 --cache 3`
- Muốn **test vừa vừa** → dùng `--samples 300 --cache 300`
- Muốn **reuse cache 500 đã có** → dùng `--cache 500`
- Muốn **chạy full production** → dùng `--cache full`

---

## 8) File cache tương ứng

- `--cache 3` → `data/cache/triples_3.json`
- `--cache 300` → `data/cache/triples_300.json`
- `--cache 500` → `data/cache/triples_500.json`
- `--cache full` → `data/cache/triples_full.json`

Ngoài ra `data/cache/triples.json` vẫn là alias mặc định để tương thích code cũ.

### File graph tương ứng

- `--cache 3` → `data/processed/kg_3.pkl`
- `--cache 300` → `data/processed/kg_300.pkl`
- `--cache 500` → `data/processed/kg_500.pkl`
- `--cache full` → `data/processed/kg_full.pkl`

Ngoài ra `data/processed/kg.pkl` vẫn được cập nhật như alias mặc định để code cũ không bị vỡ.

---

## 9) Output / reports nằm ở đâu

Sau khi dọn workspace, các file output được gom về:

- `reports/full_pipeline/` → `full_pipeline_results_*.json`
- `reports/evaluations/` → `quick_quality_eval_*.json`, `eval_passages*.txt`, `eval_result*.txt`
- `reports/benchmarks/` → `bench_*.json`
- `reports/logs/` → các file `.log`, `full_run_err.txt`, `full_run_log.txt`, `test_output.txt`
