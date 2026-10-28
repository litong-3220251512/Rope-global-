# CLEVR 开源模型架构与可比性核查

核查日期：2026-10-24。适用协议：[PUB-R1.1](rope_public_data_experiment_plan.md)。

## 1. 结论与使用边界

本文件回答两个问题：PUB-R1.1 的公共模型具体是什么，以及哪些已发表、可定位源码的架构可作为实现或诊断参考。它不是排行榜，也不把不同输入、监督、预训练、数据划分或训练预算下的数字拼成同条件比较。

截至核查日期，本文核读的论文中，没有一篇报告以下完整条件：CLEVR 官方 RGB 与原始问题文本、PUB-R1.1 的单关系存在性程序筛选、scene-level 确定性划分、二分类、五个配对初始化 seed 和每 run 8000 次更新。因此，目前没有证据支持把任何候选称为该任务的“同条件 SOTA”。下文使用“开源强架构参考”或“候选”，不使用无条件 SOTA 表述。论文中的完整 CLEVR `Exist` 类别还包含 PUB-R1.1 筛选规则之外的问题，不能当作本任务准确率。

核查得到的直接结论是：

- **RoPE-ViT** 是 G 与 M10 最接近的二维旋转位置编码实现来源，但论文实验是图像分类，不是 CLEVR 问答。
- **FiLM 的从像素训练版本**在输入和监督类别上最接近本任务：RGB、原始问题和答案监督，不依赖对象框、场景图或程序标签；其卷积骨干、显式坐标通道、完整 CLEVR 任务和训练协议仍与 PUB-R1.1 不同。它目前只是尚未排序、不可直接实施的文献候选。
- **MAC 与 LCGN**提供问题引导的多步推理或网格消息传递参考，但其论文配置使用 ImageNet 预训练视觉特征，不能作为当前随机初始化 ViT 的同条件性能基线。
- **Relation Network（RN）**有适用的从像素关系推理定义，但本次未核实到作者发布的 CLEVR 实现，不能列入“可直接复用的官方开源实现”。
- **NS-CL 与 MDETR**的发布接口或论文训练条件使用对象框、程序、场景图派生对齐等额外监督，不满足 PUB-R1.1 的 answer-only 模型边界，不能原样接入。

本文件不授权实现或训练候选模型。当前实施顺序仍以 PUB-R1.1 第 10 节为准：先完成公共 RGB+问题骨干以及 G/B、M10、Q-only、配对训练和统计报告；只有 RGB 模型相对 Q-only 与训练多数类基线建立可重复任务增量后，才讨论新的位置编码候选。

## 2. PUB-R1.1 公共架构

### 2.1 输入、张量形状与信息流

对 batch 大小 \(N\)，图像输入为 \(X\in\mathbb{R}^{N\times3\times224\times224}\)。`Conv2d(3, 192, kernel_size=16, stride=16)` 产生 \(14\times14=196\) 个 patch token：

\[
X\longmapsto Z\in\mathbb{R}^{N\times196\times192}.
\]

原始问题按 PUB-R1.1 第 4.2 节编码为变长 token 序列。`Embedding(vocab_size, 64, padding_idx=0)` 后接单层、单向 `GRU(input_size=64, hidden_size=192)`；取包含 `<END>` 的最后一个有效状态 \(h\in\mathbb{R}^{N\times192}\)，而不是 padding 后状态。令 \(c\in\mathbb{R}^{192}\) 为可训练 CLS 向量，则

\[
\mathrm{CLS}_0=c+h.
\]

视觉与语言只在 CLS 初始化处融合。拼接后的序列形状为 \(N\times197\times192\)，再经过六层 Pre-LN Transformer。每层使用 3 个 attention head，`head_dim=64`，MLP 隐层宽度为 768，激活为 GELU；LayerNorm epsilon 为 \(10^{-6}\)，LayerScale 初值为 \(10^{-4}\)，dropout 与 drop-path 均为 0。输出为

\[
[\mathrm{CLS}_0;Z]
\longmapsto \mathrm{LN}(\mathrm{CLS}_6)
\longmapsto \mathrm{Linear}_{192\to2}
\longmapsto (\ell_{no},\ell_{yes}).
\]

模型 batch 只能包含 RGB、问题编码、真实长度和监督答案。程序、场景图、对象索引、对象坐标、关系表、筛选编码、文件编号和划分审计字段均不得进入模型前向。

### 2.2 四个正式组只允许的差别

| 组别 | 位置结构或视觉输入 | 相对公共骨干的新增参数 | 允许解释 |
| --- | --- | ---: | --- |
| G | 所有 patch Q/K 使用 base100 的固定二维 Axial RoPE；CLS 不旋转 | 0 | 主基线 |
| B | 每轴后 8 个低频平面使用 2×2 区域内坐标，固定 \(\lambda=1\)；其余与 G 相同 | 0 | 唯一主比较 B−G |
| M10 | 每层、每头学习 32 个二维频率向量，base10 初始化 | 1152 | 已有 Mixed 方法的整体参照；不隔离频谱、方向与可学习性 |
| Q-only | 保留 G 的位置结构，以 196 个零 token 绕过 patch projection，且不读取图像 | 0 | 问题先验与非视觉计算对照 |

G/B 的二维平面顺序为每头前 16 个 x 平面、后 16 个 y 平面。只旋转 patch 的 Q/K；CLS 与 V 不旋转。B 的局部坐标替换保持同一区域内指定 patch 对的相对相位，但不推出整层或整网输出等价。完整定义、边界例和容差见 [PUB-R1.1 第 3、6 节](rope_public_data_experiment_plan.md#3-位置变换的数学契约)。

该公共架构是本项目冻结的实验合同，不是某篇论文已验证的 CLEVR 最优结构，也没有尚未运行的本地性能结果。

## 3. 可比性判据

候选架构按以下字段核查。任一字段不同，都应在独立实验中控制或明确标注，不能直接按论文准确率排序。

| 判据 | PUB-R1.1 条件 |
| --- | --- |
| 模型可见输入 | 官方 RGB + 原始问题 token；不含程序、对象、场景图、坐标真值或文件 ID |
| 训练监督 | 入选问题的二元答案 |
| 数据任务 | 固定程序筛选后的单关系存在性问题；先按 scene/image 划分再筛选 |
| 视觉初始化 | 公共非 PE 参数随机初始化并在 G/B/M10/Q-only 间逐 tensor 配对 |
| 主干与容量 | 六层、宽度 192 的 ViT；G/B 参数量相同 |
| 训练预算 | 五个 seed，每 run 固定 8000 次 optimizer 更新 |
| 主要估计量 | 基础测试微平均准确率的配对差 \(\Delta_s=\mathrm{Acc}_{B,s}-\mathrm{Acc}_{G,s}\) |
| 进入实现前的复用门槛 | 论文正文与作者/机构官方仓库均可定位；固定 revision，并记录许可证、维护状态、所需 API 及采用或不采用的理由 |

“RGB + question”本身不足以证明可比。例如，固定 ImageNet 特征、预训练语言模型、对象框监督或程序派生对齐都会改变可用信息或优化难度；完整 CLEVR 28/29 类答案分类也不是当前二分类筛选任务。

该表最后一行是进入实现前必须满足的门槛，不表示下文所有文献候选已经通过复用审查。本项目当前实际采用的 RoPE-ViT 已固定 commit；FiLM、MAC、LCGN、NS-CL、MDETR、VisionLLaMA、LieRE 与 PoPE 本轮只核读论文和官方仓库中的相关接口，尚未统一固定 revision 或验证运行环境。它们若被提议进入实现，须先另做完整的版本、许可证、维护状态和 API 记录。论文级排除项 RN 不进入当前库复用审查。

## 4. 最接近的位置编码实现：RoPE-ViT

**论文与源码。** Heo et al., *Rotary Position Embedding for Vision Transformer*, ECCV 2024：[论文正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf)、[官方仓库](https://github.com/naver-ai/rope-vit)。本项目固定的上游版本为 commit [`48d8df5`](https://github.com/naver-ai/rope-vit/tree/48d8df50250056a837672609bf764d80fb545869)，以 Git submodule 保留。

**论文定义。** 论文第 3.1 节式 (7)–(11) 将旋转施加于 attention 的 query 和 key；第 3.2 节式 (12)–(13) 定义固定 Axial 频率，式 (14) 定义 Mixed 相位

\[
\phi_{n,t}=\theta_t^x p_n^x+\theta_t^y p_n^y,
\]

并令二维频率按层、按头学习；式 (15) 给出对应的相对位置交互。论文第 4.1 节与 Figure 4 的证据是 ImageNet-1k validation top-1，使用 ViT-S/B/L、224×224 训练输入和 DeiT-III 400-epoch 配方；这不能支持 CLEVR 问答性能断言。

**源码信息流。** 官方 [`models/vit_rope.py`](https://github.com/naver-ai/rope-vit/blob/48d8df50250056a837672609bf764d80fb545869/models/vit_rope.py) 中：

- `compute_axial_cis` 生成固定二维 Axial 相位；
- `init_random_2d_freqs` 与 `compute_mixed_cis` 构造按层、按头的 Mixed 频率及相位；
- `apply_rotary_emb` 将实数相邻维重解释为复数对后旋转；
- `RoPEAttention.forward` 只旋转 patch Q/K，保留 CLS 和 V；
- `rope_vit_models.forward_features` 完成 patch embedding、CLS 拼接和 Transformer 主干。

对 \(N\) 个样本、\(H\) 个头、\(T=196\) 个 patch 和每头宽度 \(d=64\)，Q/K/V 在 attention 中可写为 \(N\times H\times(T+1)\times d\)；旋转作用于其中的 \(T\) 个 patch 位置。PUB-R1.1 的 G 沿用该数据流但锁定 base100；M10 对应官方 Mixed 参数化的受控小模型映射；B 只替换一半低频平面的坐标，不增加频率参数。

**许可证与复用边界。** 仓库根部的 `LICENSE_rope-vit` 是 Apache-2.0，但 `NOTICE` 明确列出 `models/vit_rope.py` 等文件源自 CodeLlama，并附带 Llama 2 条款。因此，不能只依据 README 徽章把所有文件概括为无例外 Apache-2.0；任何复制或改写须同时检查具体文件与 `NOTICE`。当前项目通过固定 submodule 与薄适配保留来源，不把上游 ImageNet 结果当成本地问答结果。

## 5. 输入与监督最接近的开源问答架构

### 5.1 FiLM：从像素训练的待定义诊断参考

**论文与源码。** Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018：[论文正文](https://arxiv.org/pdf/1709.07871)、[作者仓库](https://github.com/ethanjperez/film)。

**核心计算。** 论文第 2.1 节式 (1)–(2) 定义问题条件化仿射调制。对第 \(i\) 个 feature map，

\[
\operatorname{FiLM}(F_{i,c}\mid\gamma_{i,c},\beta_{i,c})
=\gamma_{i,c}F_{i,c}+\beta_{i,c}.
\]

第 2.2 节与 Figure 3 给出完整流：问题词嵌入进入 GRU，GRU 生成四个 FiLMed residual block 的 γ/β；RGB 进入卷积视觉前端，空间坐标通道与视觉 feature map 拼接；四个 block 逐层调制后，经 pooling 与 MLP 输出 CLEVR 答案。论文同时报告固定 ImageNet ResNet-101 特征版和从 RGB 像素训练的 CNN 版。

**官方实现定位。** [`vr/models/film_gen.py`](https://github.com/ethanjperez/film/blob/master/vr/models/film_gen.py) 的 `FiLMGen` 负责问题编码与调制参数生成；[`vr/models/filmed_net.py`](https://github.com/ethanjperez/film/blob/master/vr/models/filmed_net.py) 的 `FiLM`、`FiLMedResBlock` 与 `FiLMedNet` 负责逐通道调制、残差视觉计算和答案分类。

**结果边界。** 论文 Table 1 报告完整 CLEVR 的总体准确率：预训练特征版 97.7%，从像素版 97.6%；该表及相邻正文未明确说明 split，因此这里将其 split 记为未知。Table 2 明确是 validation ablation，并给出 raw-pixels 97.6%。这些数字对应完整多答案任务和论文配方，不是 PUB-R1.1 的二分类 test，也没有 G/B 配对控制。

**为何值得进一步定义。** 从像素版只依赖 RGB、问题和答案监督，在信息类别上最接近 PUB-R1.1，可能用于检查当前 ViT 的任务学习是否受视觉—语言融合路径限制。但这只是待验证解释：FiLM 的卷积归纳偏置、显式 x/y 坐标、条件化位置、容量和优化都与当前模型不同；即使它在相同筛选数据上学会任务，也不能证明 B 的区域相位机制有效，也不能单独归因于 FiLM 调制。

**工程状态。** 官方代码许可证为 [CC BY-NC 4.0](https://github.com/ethanjperez/film/blob/master/LICENSE)，且 [`requirements.txt`](https://github.com/ethanjperez/film/blob/master/requirements.txt) 仍引用 PyTorch 0.1.11。该仓库适合核对结构，不应未经许可证与现代 PyTorch 兼容性审查直接复制。

### 5.2 MAC：多步控制—读取—写入

**论文与源码。** Hudson and Manning, *Compositional Attention Networks for Machine Reasoning*, ICLR 2018：[论文正文](https://arxiv.org/pdf/1803.03067)、[作者仓库](https://github.com/stanfordnlp/mac-network)。

**核心计算。** 论文第 2.1–2.3 节与 Figures 2–3 将问题编码、知识库和答案头连接到重复的 MAC cell。每个 cell 依次执行：control unit 从问题词表示中选择当前推理步骤；read unit 用 control 和上一记忆从视觉知识库读取；write unit 将读出结果写入新记忆。论文默认配置使用 12 个推理步骤、状态宽度 512、问题 biLSTM，以及固定 ImageNet-pretrained ResNet-101 conv4 feature 后的两个可训练卷积层；最终按 28 个答案做交叉熵分类。

**官方实现定位。** [`mac_cell.py`](https://github.com/stanfordnlp/mac-network/blob/master/mac_cell.py) 的 `MACCell.control`、`read`、`write` 对应三个单元；[`model.py`](https://github.com/stanfordnlp/mac-network/blob/master/model.py) 的 `MACnet` 连接问题 encoder、视觉 feature、循环推理和输出层。

**结果边界。** 论文 §4 报告完整 CLEVR validation accuracy 98.94%，Appendix Table 2 在 validation ablation 中给出 98.9%。该结果使用预训练视觉特征、完整多答案任务和不同训练预算，不能与 PUB-R1.1 排名。

**工程状态。** 官方代码为 [Apache-2.0](https://github.com/stanfordnlp/mac-network/blob/master/LICENSE)，README 说明原实现基于 TensorFlow 1.3。它适合作为多步问题引导推理的架构参考，不是当前 PyTorch 主干的可直接替换件。

### 5.3 LCGN：语言条件化的网格消息传递

**论文与源码。** Hu et al., *Language-Conditioned Graph Networks for Relational Reasoning*, ICCV 2019：[论文正文](https://arxiv.org/pdf/1905.04405)、[作者仓库](https://github.com/ronghanghu/lcgn)、[官方 PyTorch 分支](https://github.com/ronghanghu/lcgn/tree/pytorch)。

**核心计算。** 论文第 3.1 节式 (1)–(8) 与 Figure 2 定义语言条件化消息传递：每轮先从问题表示提取 textual command，再以 command 调制网格节点之间的消息与聚合；第 3.2 节式 (9)–(10) 定义 single-hop answer prediction。CLEVR 配置使用 \(14\times14\times1024\) 的 ImageNet-pretrained ResNet-101 C4 网格、问题 biLSTM 和四轮消息传递；Appendix A 说明将 sinusoidal spatial encoding 拼接到网格 feature。该 CLEVR 配置不使用对象 proposal、场景图或程序监督。

**官方实现定位。** TensorFlow 主分支中，[`models_clevr/input_unit.py`](https://github.com/ronghanghu/lcgn/blob/master/models_clevr/input_unit.py) 编码问题，[`models_clevr/lcgn.py`](https://github.com/ronghanghu/lcgn/blob/master/models_clevr/lcgn.py) 的 `extract_textual_command` 与 `propagate_message` 实现语言条件化传播，[`models_clevr/model.py`](https://github.com/ronghanghu/lcgn/blob/master/models_clevr/model.py) 与 [`output_unit.py`](https://github.com/ronghanghu/lcgn/blob/master/models_clevr/output_unit.py) 完成 single-hop 分类。

**结果边界。** 论文 §4.1/Table 3 报告 official CLEVR test accuracy 97.9%。官方 README 报告 97.90% validation、97.88% test，并说明提供的 PyTorch checkpoint 从 TensorFlow 移植。预训练视觉特征、sinusoidal coordinates、消息传递容量和完整任务均与 PUB-R1.1 不同。

**工程状态。** 仓库采用 [BSD-2-Clause](https://github.com/ronghanghu/lcgn/blob/master/LICENSE.txt)；主 README 的原始环境是 TensorFlow 1.12，虽有官方 PyTorch 分支，但本项目尚未验证两者训练等价。LCGN 可用于解释“密集网格上的显式关系传播”这一竞争机制，不应直接替换 G/B 后把收益归因于位置编码。

## 6. 论文相关但当前不直接采用的架构

| 架构 | 已核实的论文条件 | 不直接采用的原因 |
| --- | --- | --- |
| Relation Network (RN) | Santoro et al., NeurIPS 2017，[§2 式 (1)、§4 Figure 2](https://proceedings.neurips.cc/paper_files/paper/2017/file/e6acf4b0f69f6f6e60e9a815938aa1ff-Paper.pdf)：四层 scratch CNN 产生带坐标的空间对象，LSTM 编码问题，\(g_\theta\) 遍历对象对后求和，再由 \(f_\phi\) 分类。Table 1 报告 official CLEVR test 95.5% overall、97.8% Exist；论文 §4 的答案头为 29 类。 | 本次未核实到作者发布的 CLEVR 代码、许可证和维护状态。论文定义可作数学参考，但不满足“官方开源实现可定位”的复用门槛。 |
| NS-CL | Mao et al., ICLR 2019，[§3.1 Figures 2–4、§3.2 式 (1)](https://arxiv.org/pdf/1904.12584)：proposal、概念表示、语义 parser 与准符号 executor。§4 使用 4000 张生成 CLEVR 图像的 bounding-box 监督预训练 detector；官方 [README](https://github.com/vacancy/NSCL-PyTorch-Release#training-and-evaluation) 说明发布版评估需要预解析程序和检测框。 | 发布接口直接消费对象/程序，且训练条件引入额外 detector 监督与生成 curriculum；违反 PUB-R1.1 的模型输入和 answer-only 监督边界。 |
| MDETR | Kamath et al., ICCV 2021，[§2.2.1/Figure 2 与 Appendix B](https://ar5iv.labs.arxiv.org/html/2104.12763)：ResNet 图像特征与 DistilRoBERTa 文本特征进入联合 Transformer；CLEVR 训练用 scene graph 重建框、用 functional program 派生对象—文本对齐。Table 1/Appendix Table 7 报告 official test 99.7% overall、99.9% Exist。 | 推理虽接收 RGB+文本，但论文训练使用额外框与对齐监督、视觉和语言预训练以及两阶段训练；数字不能支持 answer-only 的 PUB-R1.1 比较。 |

以上排除是针对“原样作为当前基线”，不是对方法质量的否定。若未来改变监督或预训练边界，必须升版协议并把新增信息、容量与预算作为独立因素控制。

## 7. 其他视觉位置编码参考

这些工作有助于定义位置编码对照，但没有提供与 PUB-R1.1 相同的 CLEVR 问答结果。

| 方法 | 论文与实现中的机制 | 与当前任务的关系 |
| --- | --- | --- |
| VisionLLaMA | Chu et al., ECCV 2024，[§3.1–3.3、式 (1)–(4)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08280.pdf)；[官方实现](https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/deit/my_models.py)含 `precompute_freqs_cis_2d`、`apply_rotary_emb` 与 `RAttention`。AS2DRoPE 按 anchor/current resolution 缩放坐标。 | 可核对固定二维旋转与分辨率缩放；完整采用还会改变 SwiGLU、pooling 或 attention topology。仓库根部的统一复用许可证本次未核实，暂不复制源码。 |
| LieRE | Ostmeier et al., ICML 2025，[§4/Figure 3](https://arxiv.org/html/2406.10322v5#S4)：\(R(p)=\exp(\sum_a p_aA_a)\)，其中 \(A_a^\top=-A_a\)；[官方实现](https://github.com/StanfordMIMI/LieRE/blob/main/models/rope_vit.py)用 `torch.matrix_exp` 学习旋转。 | 引入按层/头的矩阵生成元和额外计算；非交换生成元下不自动保留普通 RoPE 的精确相对坐标性质。属于新的可学习 PE 容量，不是 B 的等容量替代。 |
| PoPE | Gopalakrishnan et al., ICML 2026，[§3 式 (3)–(6)](https://arxiv.org/html/2509.10534v3#S3)将 content 表示为正幅值并以位置相位计算 attention；[官方代码](https://github.com/agopal42/pope)实现因果序列模型。 | 已核实实验覆盖若干一维因果序列任务，包括 indirect indexing、符号音乐、基因组和语言建模；尚未核实二维双向视觉实验。发布实现没有可直接采用的二维双向视觉映射；“polar”指 feature 的极坐标表示，不是图像极坐标。 |

这三项若进入实验，都需要先给出二维坐标、CLS 行为、参数量、初始化、计算增量和退化回 G 的完整数学定义，再单独审批预算。现阶段 M10 已覆盖“官方 RoPE-ViT 可学习 Mixed 频率”这一最近对照，不应同时堆叠多个新 PE 变体。

## 8. 历史数字为何不能组成排行榜

下表只记录论文报告及其条件，便于防止后续误引；它不提供同条件名次。

| 方法 | 论文报告 | 输入/监督或 split 条件 | 对 PUB-R1.1 的含义 |
| --- | --- | --- | --- |
| FiLM | Table 1：97.7% pretrained features、97.6% raw pixels，split 未明确；Table 2：97.6% raw pixels，明确为 validation ablation | 完整 CLEVR、多答案分类、RGB+问题+答案；内部坐标通道 | 输入/监督类别接近，任务与架构不同 |
| MAC | 98.9% validation，Appendix Table 2 | 固定 ImageNet ResNet-101 feature；完整 CLEVR | 有预训练与容量差异 |
| LCGN | 97.9% official test，Table 3 | ImageNet ResNet-101 C4；完整 CLEVR | 有预训练、空间编码和推理结构差异 |
| RN | 95.5% overall、97.8% Exist，official test，Table 1 | scratch CNN；完整 CLEVR；29 类答案头 | Exist 范围大于当前筛选，且官方代码未核实 |
| NS-CL | 98.9% official validation（小数据），Table 4；99.2% validation（完整训练设置），Table 3/Appendix F.3 | detector 框监督、curriculum；不同设置不可混合 | 超出 answer-only 边界 |
| MDETR | 99.7% overall、99.9% Exist，official test，Table 1/Appendix Table 7 | 预训练视觉/语言编码器；scene graph/program 派生监督 | 超出输入与监督边界 |

这些数字来自不同年份、架构、split、答案空间、训练数据量、预训练和选择规则。即使数值更高，也不能说明某方法在 PUB-R1.1 上更优；没有本任务的原始预测，也不能复算当前指标或配对差。

## 9. 若公共骨干未学会任务，如何使用这些参考

在获得用户授权的运行结果前，不预判公共骨干会失败。若未来出现“G 与 M10 均接近 Q-only 或训练多数类”的现象，处理顺序应是：

1. 先检查数据、标签复算、问题长度、图像通路、梯度、恢复状态和共同初始化，不把工程错误解释为架构不足。
2. 若实现正确但任务增量仍不可重复，记录“当前配方不足以定位 PE 效应”，不宣布 RoPE 或 B 无效。
3. 若要提出新的架构诊断，在独立协议中可继续评估 **scratch FiLM** 是否值得进入候选集，但本文件不对候选排序，也不构成实施定义。至少还缺少精确 feature-map 与调制张量形状、参数/FLOPs 增量、初始化、退化回无 FiLM 条件的方式、预注册证伪判据，以及分别控制 CNN、显式坐标、FiLM 调制和容量的消融。在这些要素补齐并获批前不得编码。未来若形成合格方案，其输出须为二分类并使用相同 train/dev/test manifest；该实验只诊断另一种视觉—语言架构能否学习任务，不参与 B−G 主比较。
4. 只有当诊断需要区分单步融合与关系推理容量时，再考虑 MAC 或 LCGN。若采用预训练 feature，应与随机初始化条件分表报告；若改为 scratch RGB，则这是新的实现，不得引用论文原始准确率作为其验证。
5. 任一候选在编码前先冻结：输入输出形状、监督、坐标、参数与 FLOPs 增量、初始化、预算、退化条件、失败预测和逐项消融。一次只验证一个主要机制。

因此，当前最小工程决策不是新增一个“更先进”的模型，而是完成 PUB-R1.1 已冻结的 G/B/M10/Q-only 链路。以上架构用于解释失败和设计后续对照，不用于提前改变主实验。

## 10. 证据状态摘要

| 断言 | 证据类型与位置 | 适用范围 | 相反证据或未知项 | 允许表述 |
| --- | --- | --- | --- | --- |
| RoPE-ViT 定义 Axial 与 Mixed 视觉旋转 | 论文 §3.1–3.2 式 (7)–(15)；官方 `models/vit_rope.py` | 图像 Transformer 的 Q/K 位置编码 | 论文未评估当前 CLEVR 任务 | “最接近的 PE 实现来源” |
| scratch FiLM 可从 RGB+问题+答案训练完整 CLEVR | 论文 §2.1–2.2、Tables 1–2；官方 `film_gen.py`、`filmed_net.py` | 完整 CLEVR、多答案分类 | 当前筛选、split、预算和配对 PE 未评估 | “输入/监督类别最接近的诊断参考” |
| MAC/LCGN 在完整 CLEVR 上报告高准确率 | MAC §4/Appendix Table 2；LCGN §4.1/Table 3 | 各自论文的预训练视觉条件 | 与当前随机初始化二分类条件不一致 | “强架构参考”，不称同条件基线 |
| RN 有从像素对象对聚合定义 | RN §2 式 (1)、§4 Figure 2 | 论文架构与 official test 报告 | 作者 CLEVR 代码与许可证未核实 | “论文参考，官方实现未核实” |
| NS-CL/MDETR 原配方超出 answer-only 边界 | NS-CL §4 与发布 README；MDETR Appendix B.1–B.2 | 各自发布训练与接口 | 未评估移除额外监督后的版本 | “不适合原样接入” |
| 存在 PUB-R1.1 同条件 SOTA | 未找到支持证据 | 无 | 检索不到不等于不存在；当前核查范围有限 | “尚未找到，不能声称” |

后续若出现新论文、官方仓库更新或本地实验结果，应追加来源、条件和更正记录；不得静默把历史数字改写为当前任务证据。
