# domain-tune-lab

这是一个重新开发的 LoRA 微调实验项目，目标是做出可以量化证明提升的结果，而不是继续做开放式客服生成。

当前任务选择公开中文情感分类数据集 ChnSentiCorp：输入一条中文评论，模型只能输出 `正面` 或 `负面`。这个任务有标准答案，因此可以直接比较 Base 原模型和 LoRA 微调模型的准确率、宏 F1、格式合规率。

## 为什么换成情感分类

开放式客服回复很难证明“变好”，因为同一个问题可以有很多种合理回答。ChnSentiCorp 是二分类任务，评测标准清楚：

- `Accuracy`：预测是否正确。
- `Macro F1`：正负样本是否都学到。
- `Format Valid`：是否严格只输出 `正面` 或 `负面`。
- `Error Cases`：保留错误样例，方便复盘。

## 项目结构

```text
domain-tune-lab/
  configs/
    qwen2.5_0.5b_chnsenticorp_lora_smoke.yaml
    qwen2.5_0.5b_chnsenticorp_lora.yaml
    llamafactory_dataset_info.json
  scripts/
    run_wsl_setup.sh
    train_smoke.sh
    train_full.sh
    evaluate.sh
  src/domain_tune_lab/
    prepare_chnsenticorp.py
    sync_llamafactory_data.py
    evaluate_sentiment_models.py
```

## 运行步骤

在 WSL2 Ubuntu 中执行：

```bash
cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
bash scripts/run_wsl_setup.sh
```

如果提示缺少 `python3-venv`，先执行：

```bash
bash scripts/install_wsl_system_deps.sh
```

如果 `/mnt/d/home/work/tools/LLaMA-Factory` 还没有安装：

```bash
mkdir -p /mnt/d/home/work/tools
cd /mnt/d/home/work/tools
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
source /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab/.venv-wsl/bin/activate
pip install -e ".[torch,metrics]"
```

先跑 smoke test：

```bash
cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
bash scripts/train_smoke.sh
```

简历级快速实验训练，约使用 2000 条公开训练样本：

```bash
bash scripts/train_resume_grade.sh
```

全量训练：

```bash
bash scripts/train_full.sh
```

评测：

```bash
bash scripts/evaluate.sh
```

评测输出：

```text
outputs/chnsenticorp_eval/eval_base.jsonl
outputs/chnsenticorp_eval/eval_lora.jsonl
outputs/chnsenticorp_eval/metrics.json
outputs/chnsenticorp_eval/eval_report.md
```

启动前端测试台：

```bash
bash scripts/serve_ui.sh
```

默认地址：

```text
http://localhost:7861/
```

## 当前实验结果

已完成一次简历级快速实验：

- 训练数据：公开 ChnSentiCorp train split 中 1928 条有效样本。
- 验证数据：公开 ChnSentiCorp valid split 中 391 条有效样本。
- 测试数据：公开 ChnSentiCorp test split 全量 1179 条。
- 基座模型：`Qwen/Qwen2.5-0.5B-Instruct`。
- 微调方法：4bit QLoRA，LoRA 可训练参数约 879.8 万，占总参数 1.75%。
- 输出 adapter：`checkpoints/qwen2.5-0.5b-chnsenticorp-lora`。

| 模型 | Accuracy | Macro F1 | Format Valid | Avg Latency |
| --- | ---: | ---: | ---: | ---: |
| Base | 68.96% | 66.57% | 100.00% | 0.07s |
| LoRA | 92.20% | 92.20% | 100.00% | 0.13s |

结论：在公开测试集上，LoRA 相比 Base 的 Accuracy 提升 23.24 个百分点，Macro F1 提升 25.62 个百分点。Base 明显偏向预测 `负面`，LoRA 后正负样本召回更均衡。

## 前端效果

同一句“有缺点但整体值得购买”的评论，LoRA 能判断为 `正面`，Base 原模型容易被局部负面词误导为 `负面`。

![中文情感 LoRA 前端对比](docs/assets/sentiment-lora-ui-comparison.png)

## 简历表达

可以写成：

> 基于 Qwen2.5-0.5B-Instruct 和公开 ChnSentiCorp 数据集构建中文情感分类 LoRA 微调系统，完成公开数据转换、QLoRA 训练、Base/LoRA 自动化评测与错误样例分析；在 1179 条公开测试集上将 Accuracy 从 68.96% 提升到 92.20%，Macro F1 从 66.57% 提升到 92.20%。

## 数据说明

训练、验证、测试数据均从公开 ChnSentiCorp 数据集转换而来。项目不再使用自建客服规则数据，也不混入人工构造答案。

## GitHub 发布建议

建议提交到 GitHub 的内容：

- `src/`：数据准备、训练、评测、推理服务代码。
- `frontend/`：Base / LoRA 对比测试页面。
- `scripts/`：环境、训练、评测、前端启动脚本。
- `configs/`：保留训练配置，方便说明实验参数。
- `outputs/chnsenticorp_eval/eval_report.md`：保留最终评测报告。
- `README.md`、`pyproject.toml`、`requirements-wsl.txt`。

不建议直接提交：

- `.venv-wsl/`：本地虚拟环境，体积很大。
- `data/processed/**/*.jsonl`：由公开数据脚本生成，不需要重复分发。
- `checkpoints/`：模型 adapter 和训练中间 checkpoint 较大，建议用 Git LFS、GitHub Release 或 Hugging Face 单独发布。
- `outputs/**/*.jsonl`：逐条预测结果较冗余，保留 Markdown 报告即可。

如果要把 LoRA adapter 也放进 GitHub，推荐使用 Git LFS：

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes checkpoints/qwen2.5-0.5b-chnsenticorp-lora/adapter_model.safetensors
```
