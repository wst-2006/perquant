# A10G 24G GPU 全量训练操作手册

## 目标

在租来的 GPU 机器上离线运行第一轮 ML baseline，不连接本机数据库，不上传 `.env`、token、密码。

## 1. 本机打包

在本机项目目录运行：

```powershell
python scripts/package_gpu_experiment.py --output-path reports/ml_lab/gpu_experiment_package.zip
```

上传到 GPU 的文件：

- `reports/ml_lab/gpu_experiment_package.zip`

不要上传：

- `.env`
- 数据库目录
- Tushare token
- 数据库密码
- 券商账户文件

## 2. 租 GPU

推荐：

- `A10G 24G`
- 租期：`3 天`
- Ubuntu 20.04/22.04
- conda + cuda + pytorch/jupyter 镜像均可

## 3. 上传与解压

把 `gpu_experiment_package.zip` 上传到 GPU 机器，例如上传到 `/root/quant_gpu/`。

在 GPU 终端执行：

```bash
cd /root/quant_gpu
unzip gpu_experiment_package.zip
```

## 4. 检查文件

```bash
ls
ls reports/ml_lab
python --version
nvidia-smi
```

应看到：

- `src/`
- `scripts/`
- `requirements-gpu.txt`
- `reports/ml_lab/training_frame_full.csv`
- `reports/ml_lab/experiment_manifest.json`

## 5. 运行全量 baseline

```bash
bash scripts/run_gpu_full_baseline.sh
```

预计：

- A10G 24G：约 4-8 小时
- 慢时可能 8-12 小时

不要关闭实例。终端断开时如果平台不保留进程，建议用平台自带 Jupyter/终端保持运行。

## 6. 跑完下载

下载这些文件/目录：

- `reports/ml_lab/research_harness_v1_full/research_harness_report.json`
- `reports/ml_lab/research_harness_v1_full/research_harness_summary.csv`
- `logs/gpu_research_full_*/research_harness_full.log`
- `logs/gpu_research_full_*/run_status.json`
- `logs/gpu_research_full_*/requirements_gpu_lock_before.txt`
- `logs/gpu_research_full_*/requirements_gpu_lock_after.txt`
- `logs/gpu_research_full_*/nvidia_smi_before.txt`
- `logs/gpu_research_full_*/nvidia_smi_after.txt`

## 7. 清理 GPU

确认下载完成后：

```bash
rm -rf /root/quant_gpu
```

然后释放实例。

## 8. 结果判断

结果下载回本机后再审查：

- `stable` 看 Top10/Top20 正收益、低回撤、多 fold 稳定性。
- `aggressive` 看 Top10 收益/回撤比、强势弹性、弱势亏损控制。
- Top10 好但 Top20/Top50 差，标记高风险观察。
- 不直接上线模型。
