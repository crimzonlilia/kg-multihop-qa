# Scripts Folder

Các file chạy tay / debug / test đã được gom lại cho đỡ rối ở root.

## Cấu trúc

- `scripts/debug/` — script debug nhanh, kiểm tra pipeline, trace, relation, logic
- `scripts/tests/` — các test script ad-hoc / benchmark nhỏ
- `scripts/tools/` — utility script để check GPU, compare triples, evaluate triples quality

## Ví dụ chạy

```powershell
python scripts/debug/debug_pipeline.py
python scripts/tests/test_env.py
python scripts/tools/evaluate_triples_quality.py
```

> Các entrypoint chính vẫn để ở root cho tiện dùng:
>
> - `rebuild_graph.py`
> - `test_full_pipeline.py`
> - `check_triples.py`
