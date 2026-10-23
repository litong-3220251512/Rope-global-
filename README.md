# Global/Local RoPE Research Workspace

当前主线是在 PUB-R1.1 下使用 CLEVR 官方 RGB 与原始问题文本检验 Global/Local Axial RoPE；主比较为 G 与固定 `lambda=1` 的 B。完整定义、证据标准、阶段门槛和下一步操作见：

- [Stage 1 PUB-R1.1 实验协议](docs/rope_public_data_experiment_plan.md)
- [空间关系研究计划](docs/spatial_rope_research_plan.md)
- [Stage 1 数学定义与成熟实现选型](docs/stage1_clevr_math_and_library_review.md)
- [原始研究设想](Global_Local_RoPE_BC_Research_Plan_v2.pdf)
- [代理协作与工程规则](AGENTS.md)
- [VLN 科研工作流与 agent 职责](docs/vln_research_workflow.md)
- [科研规则的论文与 GitHub 调研依据](docs/research_workflow_sources.md)

当前结论：CIFAR-100 从头训练的单 seed 配对实验没有显示正收益（G 69.62%，B 68.94%）。CLEVR 只读数据审计已完成；下一代码单元是公共 RGB＋问题问答骨干及 G/B 接入，先验证 G。same/cross-region 不作为本轮主机制指标。正式实验、数据下载和权重下载均保持暂停，由用户控制。

核心文件：

- `src/local_rope.py`：G/B/C 相位构造；
- `src/classification.py`：RoPE-ViT 分类骨架；
- `src/segmentation.py`：后续密集预测骨架及共享 attention 工具；
- `src/clevr_programs.py`、`src/clevr_partitions.py`、`src/clevr_text.py`：冻结的问题准入、scene-level 划分和文本编码契约；
- `src/clevr_audit.py`、`run_clevr_data_audit.py`：显式本地根目录的只读 CLEVR 审计与四项 manifest 输出；
- `run_cifar_scratch.py`、`src/cifar_data.py`：已完成 Stage 0 的训练模板；
- `vendor/rope-vit/`：固定的官方 RoPE-ViT 上游代码。

现有测试仅验证实现，不代表真实数据效果：

```bash
CUDA_VISIBLE_DEVICES='' conda run -n flightgpt python -m unittest \
  tests.test_clevr_audit tests.test_clevr_text tests.test_clevr_programs \
  tests.test_clevr_partitions tests.test_clevr_schema tests.test_local_rope \
  tests.test_classification tests.test_attention_compatibility tests.test_segmentation -v
```

该命令按项目规范排除会执行旧哈希逻辑的 `tests.test_cifar_training` 和 `tests.test_cifar_data`。数据和权重不由脚本自动下载。
