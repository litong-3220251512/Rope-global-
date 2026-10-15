# Global/Local RoPE Research Workspace

当前主线是检验“分层局部坐标重置”是否能改善跨区域空间关系建模。完整定义、证据标准、阶段门槛和下一步操作见：

- [空间关系研究计划](docs/spatial_rope_research_plan.md)
- [Stage 1 数学定义与成熟实现选型](docs/stage1_clevr_math_and_library_review.md)
- [原始研究设想](Global_Local_RoPE_BC_Research_Plan_v2.pdf)
- [代理协作与工程规则](AGENTS.md)

当前结论：CIFAR-100 从头训练的单 seed 配对实验没有显示正收益（G 69.62%，B 68.94%）；下一步是 CLEVR 的 same-region / cross-region 受控诊断。正式实验、数据下载和权重下载均保持暂停，由用户控制。

核心文件：

- `src/local_rope.py`：G/B/C 相位构造；
- `src/classification.py`：RoPE-ViT 分类骨架；
- `src/segmentation.py`：后续密集预测骨架及共享 attention 工具；
- `run_cifar_scratch.py`、`src/cifar_data.py`：已完成 Stage 0 的训练模板；
- `vendor/rope-vit/`：固定的官方 RoPE-ViT 上游代码。

现有测试仅验证实现，不代表真实数据效果：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  /workspace/smbu_lt/my_envs/flightgpt/bin/python -m unittest discover -s tests -v
```

数据和权重不由脚本自动下载。下一阶段的具体交接目录会在 CLEVR schema 实现时固定。
