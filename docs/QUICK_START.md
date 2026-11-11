# 仓库快速指南

## 当前真实状态

- `web` 可启动，`/api/health`、`/api/ml/readiness`、`/api/factor/readiness` 已挂载
- `src` 中的 ML Lab 与 Factor Lab 服务存在，但完整跑通依赖真实数据库和 `TUSHARE_TOKEN`
- `tests` 当前可用，关键接口测试已通过
- `docs` 中旧完成度数字存在冲突，优先参考代码和测试

## 先跑什么

```bash
pytest tests/test_web.py tests/test_ml_lab_service.py tests/test_factor_lab_service.py -q
python scripts/run_web.py
python scripts/run_system_readiness.py
```

## 研究链路

### Factor Lab

```powershell
python scripts/run_factor_validation.py readiness
python scripts/run_factor_validation.py build-snapshots
python scripts/run_factor_validation.py build-forward-returns
python scripts/run_factor_validation.py evaluate
```

### ML Lab

```powershell
python scripts/run_ml_lab.py readiness
python scripts/run_ml_lab.py top10 --trade-date 20260601
```

## 需要先准备

- `.env`
- `TUSHARE_TOKEN`
- `DB_PASSWORD`

## 重要提醒

- 生产 baseline 不改
- 研究层单独验证
- 旧文档里的 55%、75%、90%、95% 不能直接当真
