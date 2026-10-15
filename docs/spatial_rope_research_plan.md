# Global/Local RoPE 空间关系研究计划

状态：2026-10-15 冻结为当前执行方案。此文档取代旧的 VOC pilot 与 CIFAR-100 scratch 阶段计划。

## 1. 目标和证据标准

研究问题是：在模型、初始化、训练数据、优化器和训练预算完全一致时，把一部分二维 Axial RoPE 通道改成分区内坐标，是否能稳定改善依赖空间关系的视觉任务？

这项工作必须同时回答三个问题：

1. **有效性**：B 相对全局基线 G 是否有稳定正收益？
2. **机制**：收益是否集中在局部相位真正发生变化的跨区域样本？
3. **外部效度**：合成诊断上的信号能否迁移到真实图像、指代表达和密集预测？

单个 seed、单个数据集、最好 checkpoint 或少量类别上的提升都不足以支撑论文主张。每一阶段预先固定主指标、最后一轮选择规则、切片和停止条件。

## 2. 方法的准确数学定义

对一个轴上的 patch 坐标 \(p\)，第 \(k\) 级区域原点为

\[
o_k(p)=\left\lfloor \frac{2^k p}{L}\right\rfloor\frac{L}{2^k},
\qquad p'_k=p-\lambda_k o_k(p).
\]

当前主实验固定 \(\lambda_k=1\)。Axial RoPE 的频率、Q/K 投影、复数旋转配对、V 和注意力范围保持不变；CLS token 不旋转。它仍是全局自注意力，没有窗口 mask。

| 组别 | 旋转平面分配 | 新增可训练参数 |
| --- | --- | ---: |
| G | 全部使用全局坐标 | 0 |
| B | 一半全局；一半使用 2×2 区域内坐标 | 0 |
| C | 一半全局；四分之一 2×2；四分之一 4×4 | 0 |

最关键的机制边界是：

\[
(p_i-o_i)-(p_j-o_j)=(p_i-p_j)-(o_i-o_j).
\]

- 两个位置在同一区域时，\(o_i=o_j\)，局部通道的相对相位与 G 完全相同。
- 两个位置跨区域时，区域原点差被减掉；在不同区域但局部坐标相同的位置会在局部通道发生相位别名。
- B 的全局一半仍携带全局相对位置，但局部一半**没有显式区域身份**。

因此当前 B 应称为“分层局部坐标重置”，不能称为“绝对象限锚定”“区域水印”或“对象感知位置编码”。若未来增加 region-ID phase、region embedding 或 pairwise region bias，那是一个新的 A 变体，必须单列比较，不能回写成当前 B 的性质。

实现入口：[src/local_rope.py](../src/local_rope.py)。

## 3. 已完成证据：Stage 0 CIFAR-100

已完成一组从随机初始化开始、G/B 仅 PE 不同的 CIFAR-100 配对实验：RoPE-ViT-S 主体，32×32、patch4、固定 \(\lambda=1\)。

| 指标 | G | B | B − G |
| --- | ---: | ---: | ---: |
| Internal dev Top-1 | 68.88% | 68.56% | −0.32 pp |
| Official test Top-1 | 69.62% | 68.94% | −0.68 pp |

配对图像 bootstrap 的 95% 区间为 **[−1.41, +0.05] pp**；McNemar \(p=0.074\)。把训练后的 B 在推理时切换到 \(\lambda=0\)，Top-1 仍为 68.94%；144 个预测标签变化，其中正确性交换为 38 对 38。

这说明工程链路能训练，单 seed 下没有正收益证据，而且模型对该相位变化不敏感。区间包含 0，不能把它写成已经证明 B 必然退化。CIFAR 分类也不能检验空间关系机制。

当前工作区没有该次运行目录和预测文件；上表是此前完成的核验记录，不能替代可重新审计的原始 artifact。后续正式阶段必须保留 manifest、源代码快照、checkpoint、逐样本预测和日志。

架构来源固定为 [RoPE-ViT commit `48d8df5`](https://github.com/naver-ai/rope-vit/tree/48d8df50250056a837672609bf764d80fb545869)。Stage 0 的小数据训练配方参考 [Locality Guidance / tiny-transformers commit `d2165f7`](https://github.com/lkhl/tiny-transformers/tree/d2165f74049c906b0afc9f957491960fb3c0cc8b)，但输入、patch、模型宽度和若干优化设置经过明确适配，不是原论文数字复现。

## 4. 总体推进路线

| 阶段 | 数据／任务 | 核心问题 | 进入下一阶段的门槛 |
| --- | --- | --- | --- |
| 0，已完成 | CIFAR-100 分类 | 基础训练是否正常 | 已完成；没有正收益 |
| 1 | CLEVR 受控空间诊断 | 同区域与跨区域关系是否按公式呈现不同效果 | G 基线有效；每个配对 seed 的效应方向一致；预注册的不确定性区间不跨越零效应 |
| 2 | VSR；RefCOCO 对 RefCOCO+ | 信号是否迁移到真实图像，并集中在空间表达 | 空间子集提升明显强于非空间控制，3 seeds 方向一致 |
| 3 | Cityscapes | 固定摄像机空间先验和跨区域密集预测是否受益 | 标准指标与预注册切片同时改善 |
| 4 | ADE20K、COCO instance | 是否能扩展到复杂场景 | 至少两个真实密集预测基准上有稳定收益 |
| 5 | ImageNet-1K 多分辨率 | 分类不退化和分辨率外推 | 防守性实验，不作为首个效果来源 |

ImageNet-100 不再单独安排：Stage 0 已完成基础分类检查，而它不能解决空间机制问题。任何阶段未过门槛时先停止扩张，分析失败原因；不靠搜索更多 \(\lambda\)、网格或层数制造一次性最好数字。

## 5. Stage 1：CLEVR 受控诊断

CLEVR 提供图像、对象属性、像素坐标、场景图和关系，适合从同一数据源构造可切片的空间任务。[官方页面](https://cs.stanford.edu/people/jcjohns/clevr/)

### 5.1 输入与任务

不把语言模型引入第一阶段。由场景 JSON 选取对象，给 RGB 图增加不参与标签计算的目标提示通道：

- 绝对位置任务：RGB + 1 个目标对象 marker channel，预测 2×2 象限。
- 相对位置任务：RGB + A/B 两个对象 marker channels，预测 A 相对 B 的二维方位 `upper-left / upper-right / lower-left / lower-right`。
- marker 由 scene JSON 的 `pixel_coords` 经过 patch-size 的整数除法唯一映射到一个 patch；该 patch 取 1，其余位置取 0。它不需要人为选择热图半径、标准差或判定阈值，也不得从类别标签生成任何像素。
- 同一输入接口对 G/B 完全相同。patch projection 的 4/5 通道权重属于共享随机初始化。

相对任务按 2×2 PE 网格切成：

- **same-region**：A、B 中心落在同一区域；这是近似等价性检查。
- **cross-region**：A、B 中心落在不同区域；这是主要机制指标。
- 标签由坐标差的严格符号直接确定；仅当某一轴坐标完全相等、对应方向在数学上未定义时排除该对象对，不设置 deadband。
- 每个 scene 只属于一个 split；同一 scene 生成的所有对象对必须留在同一 split。

Stage 1 不把 CLEVR 原始问题答案作为主任务。这样可以把变化归因到视觉空间编码，而不混入文本编码器和语言偏差。

### 5.2 数据划分

- 使用 CLEVR 官方 train 场景构造 train 和 internal dev，按 scene ID 固定 90/10 分割。
- 官方 val 场景只在所有模型、超参数和停止规则冻结后用于 final test。
- 从每个 split 内分别做标签、same/cross-region、对象距离和对象属性平衡。
- 保存完整的 scene IDs、对象索引对、标签和切片字段；通过直接比较这些结构化记录检查拆分，不做哈希验证。不得跨 split 重采样同一场景。

用户负责提供 CLEVR 数据。助手不下载数据或权重。预计需要的官方文件是 train/val 图像及对应 scene JSON；到位后先根据实际归档结构生成一份只读验收报告，再解包或实现 dataset。

### 5.3 模型与比较

主比较固定为 G 与 B，\(\lambda=1\)，先不加入 C：

- RoPE-ViT-S：12 blocks、384 dim、6 heads、head dim 64、MLP ratio 4、LayerScale。
- 从随机初始化训练，不加载 Stage 0、VOC 或 ImageNet 权重。
- 输入统一 resize 到 224×224，patch16，14×14 patch grid；热图同步 resize。
- 两个分支共享同一份非 PE 初始状态、每轮样本顺序、增强随机数和训练预算。
- 除 patch projection 的输入通道从 3 改为 4/5 外，不改 Transformer 主体。
- G/B 参数量必须完全相同；固定 \(\lambda\) 不进入优化器。

绝对位置任务和相对位置任务使用独立分类头及独立训练，避免多任务权重成为新的自由度。C 只在 B 通过门槛后按完全相同协议加入。

### 5.4 训练协议

在真实数据工程冒烟完成后锁定一次公共配方。初始候选值如下，正式长训前只允许根据 **G 的训练稳定性** 同步校准：

| 项目 | 候选值 |
| --- | --- |
| Seeds | 20261015、20261016、20261017 |
| Optimizer | AdamW，betas=(0.9,0.999)，weight decay 0.05 |
| Epoch | 100，最后一轮为主结果 |
| LR | batch128 时 5e-4，10 epoch linear warmup，之后 cosine 到 5e-6 |
| Batch | 128；若显存不足用梯度累积保持有效 batch |
| Regularization | drop path 0.1；不使用 mixup/cutmix，因为目标提示与关系标签不适合未经同步的混合 |
| Augmentation | 仅颜色增强；默认禁用几何增强。若使用水平翻转，必须同步翻转热图并重写关系标签 |
| Precision | BF16；G/B 同一 attention backend |
| Selection | 固定最后一轮；dev 只判断共同配方是否有效，不按 B 的表现选 epoch |

如果 G 在 train 和 dev 上都未学会任务，应共同修正配方并重新编号整组实验；不得只为 B 调参。

### 5.5 指标、切片与停止条件

主要指标：三个 seed 的 **cross-region relative test accuracy 的 B−G 配对差值均值**。

同时报告：

- 每个 seed 的 G/B 最后一轮 accuracy、loss、耗时、峰值显存；
- same-region、cross-region、absolute quadrant；
- 横向与纵向方向、2×2 区域对、对象中心距离分桶；
- 全图平移诊断：仅在不裁掉对象的范围内同步平移 RGB 和热图；
- 每个 test scene 的预测，用于 scene-level paired bootstrap；
- 三个训练 seed 的均值、标准差和所有配对差值。

Go：G 基线质量正常；B−G 在每个 seed 的方向一致；预先选定的配对不确定性区间整体位于零效应之上；分层结果表明收益并非来自单个稀有属性或区域。报告实际效应量，不另设人为百分点门槛。

No-Go：效应方向不一致、不确定性覆盖零效应、same-region 与 cross-region 没有符合推导的差异，或 G 本身未学会且在共同配方校准后仍不可靠。No-Go 后暂停现有 B/C 主线；如要研究显式 region-ID，只能以新方法 A 开立新假设与新实验表。

## 6. Stage 2–5 的预注册骨架

### Stage 2：真实空间关系与指代表达

- **VSR**：冻结相同文本编码器和融合头，只替换视觉 RoPE；按关系类别和跨网格程度报告。[论文](https://aclanthology.org/2023.tacl-1.37/)
- **RefCOCO vs RefCOCO+**：RefCOCO 的空间表达子集是目标组，减少位置词的 RefCOCO+ 是自然控制；RefCOCOg 作上下文补充。[RefCOCO/RefCOCO+](https://arxiv.org/abs/1608.00272)，[RefCOCOg](https://openaccess.thecvf.com/content_cvpr_2016/html/Mao_Generation_and_Comprehension_CVPR_2016_paper.html)
- 只有“空间表达提升强于非空间控制，且集中于跨区域目标—参照物”才支持预期机制。

### Stage 3：Cityscapes

- 使用同一公开强分割框架、相同初始化和训练预算，只替换 PE。
- 报告官方语义/实例指标，以及按实例是否跨 2×2/4×4 网格的预注册切片。
- **Boundary IoU**（真实标注边界质量）与**人工网格边界切片**必须分别报告。[Cityscapes 论文](https://openaccess.thecvf.com/content_cvpr_2016/html/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.html)

### Stage 4：ADE20K 与 COCO

- ADE20K 检查多场景语义分割；COCO 检查实例 AP、AP50、AP75、AP_S/AP_M/AP_L。
- 只有 Stage 3 通过后才确定具体框架版本和预算，防止提前绑定一个解码器造成大规模无效训练。[COCO 论文](https://vision.ics.uci.edu/papers/microsoft-coco-common-objects-in-context-2014/)

### Stage 5：ImageNet-1K 多分辨率

- 用于检查标准分类是否退化、不同输入分辨率下是否稳健。
- 不再用它寻找首个正结果，也不把分类相近解释为空间机制已经成立。

## 7. 公平性和可复现性要求

每个正式运行目录必须包含：

- 完整参数、数据文件清单、split/query manifest 和环境版本；
- G/B 共享初始状态，并用逐 tensor 精确相等或明确数值容差直接比较；
- 每轮第一批样本索引、增强参数和直接相等检查结果；
- 对应的 PR、源代码快照和逐文件 diff；
- 每轮训练/dev 日志、最后一轮 checkpoint、逐样本 final test 预测；
- 参数量、实际更新次数、训练时长、峰值显存和 GPU 标识；
- 独立重算主指标和切片的验证文件。

不得：使用哈希或摘要验证数据、代码、权重、样本和状态；按 test 选择 checkpoint；删除失败 seed；在 G/B 之间改变增强或 optimizer；用图像 bootstrap 代替训练 seed 方差；把推理时 \(\lambda=0\) 干预称为重新训练的 G。

## 8. 下一步的具体操作

按以下顺序逐步执行，每一步都先交付可检查产物：

1. **冻结 Stage 1 数据 schema**：定义 scene、对象对、单 patch marker、标签和 same/cross-region 的 JSON schema；用穷举边界案例和少量人工可读例子验证，不需要 CLEVR 文件，也不依赖人为阈值。
2. **改造模型输入**：让 patch projection 接收 4/5 通道；验证额外通道初始化在 G/B 间严格相同，现有 3 通道分类路径不回归。
3. **实现 CLEVR dataset builder**：先检索成熟的 CLEVR 数据库或 loader 并记录取舍；只写读取本地路径的代码，不包含下载行为；强制 scene-level split、类别平衡和完整 manifest。
4. **用户提供数据**：用户把官方 CLEVR 文件放到计划约定目录；助手只做尺寸、文件数、结构、schema、计数和最小可读性验收，不做哈希验证。
5. **真实数据工程冒烟**：由用户控制启动 1–2 epoch 小子集 G/B；检查 loss、resume、配对随机流、显存和 artifact 完整性。数字不进入论文表。
6. **冻结公共训练配方**：只根据 G 的可学习性决定是否共同调整候选配方，形成版本化配置。
7. **用户启动正式实验**：先完成 seed 20261015 的 G/B，再按预注册补齐另外两个 seeds；实验运行期间助手不自行下载数据、权重或占用 GPU。
8. **结果审计与分析**：用户提供完整 run 目录后，助手独立重算指标、统计 paired differences、生成曲线/切片并作 Go/No-Go 判断。

当前应执行第 1 步。正式实验和任何数据/权重下载都保持暂停，直到用户提供相应文件并明确推进到对应步骤。

第 1 步的代码前数学定义、边界条件和成熟实现选型已记录在 [`stage1_clevr_math_and_library_review.md`](stage1_clevr_math_and_library_review.md)。其中明确排除 token 级不可辨识的同 patch 对，并限定“同区域等价”只适用于指定 patch 对的局部相位，不能外推为整网输出等价。

## 9. 工作区保留策略

保留：

- `Global_Local_RoPE_BC_Research_Plan_v2.pdf`：原始研究设想；
- `src/local_rope.py`：核心 PE；
- `src/classification.py`、`src/segmentation.py`：分类与后续密集预测骨架；
- `run_cifar_scratch.py`、`src/cifar_data.py` 及对应测试：Stage 0 的可复现模板，Stage 1 runner 落地前仍有直接复用价值；
- `vendor/rope-vit/`：固定上游实现；
- `checkpoints/rope_axial_deit_small_patch16_LS.pth`：后续真实任务可能使用的公共初始化，现阶段不加载；
- 核心单元测试。

删除：旧 VOC 数据与下载归档、VOC 专用 runner/data/诊断/汇总脚本和测试、Python cache，以及已被本文吸收的旧 VOC/CIFAR 阶段文档。任何未来数据均由用户提供。

2026-10-15 已完成上述清理，释放 3,968,214,862 bytes（约 3.70 GiB）；工作区约 117 MiB。清理后 CPU 回归测试为 39 passed、1 CUDA-only skipped。正式实验、下载和 GPU 任务均未启动。

后续工程流程以仓库根目录 `AGENTS.md` 为准：数学推导与成熟库检索先于代码；禁止用人为阈值掩盖问题；先诊断根因再做有限修改；不做哈希验证；每个代码修改通过独立 PR 交付；正确运行后说明证据并准备 tag 计划。
