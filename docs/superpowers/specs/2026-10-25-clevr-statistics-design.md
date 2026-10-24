# CLEVR PUB-R1.1 统计核心设计

**状态：** 已冻结供实现；用户已授权代理按已验收计划持续推进，不逐项等待确认。  
**协议权威：** `docs/rope_public_data_experiment_plan.md`（PUB-R1.1），尤其第 7、10 节。  
**本单元目的：** 从未经挑选的基础测试逐问题预测，独立复算每 run 指标、五 seed 汇总、预注册配对差与按图像聚类 bootstrap。它不读取模型、checkpoint、RGB 或 scene，不启动训练或评估。

## 1. 意图、成功条件与边界

最终实验需要把 4 个方法、5 个正式 seed 的逐问题预测转成可审计统计，而不能只接受训练 runner 写出的汇总数。统计核心成功需要：

- 从权威审计 manifest 接收 train/base-test target，并让每个 run 保留、验证问题 ID、标签和两个 logits；图像 ID 与关系只信任 target，不信任模型输出；
- 强制正式网格 `G/B/M10/Q-only × 101..105` 完整且各 run 使用完全相同的基础测试 cohort；
- 复算微平均 accuracy、稳定二类交叉熵、四关系 × yes/no 八格、八格等权 macro；
- 复算 train-majority（只由训练标签确定，平票为 no）；
- 汇总每方法五 seed 结果，并给出逐 seed `B-G`、`M10-G`、样本 SD 与 t 区间；
- 用 seed 31001、2000 次、按图像有放回抽样的共同重数矩阵计算五 seed 平均 `B-G` 百分位区间。

非目标：文件发现、JSON/JSONL 解析与发布、训练 run 完成状态、checkpoint 选择、日志/成本汇总、GPU、训练或基础测试执行。它们属于后续独立报告/runner PR。该模块不产生性能结论；没有真实预测时只能证明实现契约。

反目标：不接受缺失 run、最好 seed、不同 cohort、按问题 bootstrap、先算每图 accuracy 再平均、按关系选择多数类、测试集多数类、非预注册切片或自动修复标签。

## 2. 已考虑的实现路径

1. **SciPy 统计分布 + NumPy bootstrap。** 优点是运行时计算 t 分位数；缺点是项目未声明 SciPy，增加依赖只为固定的一个临界值。
2. **NumPy + Python 标准库（采用）。** 用 `statistics.stdev` 算样本 SD，协议命名常量保存 `t_{4,0.975}`，用显式 `Generator(PCG64(31001))` 和 NumPy 数组做共同聚类抽样及线性分位数。不新增依赖，且与现有划分器同一 RNG 家族。
3. **自行实现 PRNG、Student-t PPF 或 quantile。** 可减少库调用，但扩大数学和复现错误面，不采用。

本单元先实现纯统计 API；预测文件读取和原子报告发布后置。纯函数边界让手算样例、错误输入和抽样不变量可直接测试，不需要伪造训练产物。

## 3. 输入、输出和维度

### 3.1 逐问题预测

一个权威 `TargetRow` 是审计 manifest 到统计层的最小投影，包含：

- `partition ∈ {train,base_test}`；train 必须来自官方 train，base_test 必须来自官方 val；
- `question_index: int >= 0`：官方问题 ID；
- `image_index: int >= 0` 与符合对应 official split 的 `CLEVR_<split>_%06d.png` 的 `image_filename`；
- `relation ∈ (left, right, front, behind)`；
- `label y_q ∈ {0,1}`，其中 no=0、yes=1。

同一 `(partition, question_index)` 不可重复。统计入口同时要求非空 train 和非空 base_test target；train targets 只决定多数类，base-test targets 是评估 cohort、图像聚类和 cell 元数据的唯一权威。

一个 `PredictionRow` 只包含：

- `question_index: int >= 0`；
- `label y_q ∈ {0,1}`，作为对写出结果时监督值的直接审计副本；
- `logits z_q=(z_{q0},z_{q1}) ∈ R^2`，两个值都必须是有限实数。

一个 `RunPredictions` 由 `method ∈ {G,B,M10,Q-only}`、`seed ∈ {101,102,103,104,105}` 和非空 row 序列组成。run 内 `question_index` 不可重复。每个 run 的 ID 集合必须与权威 base-test target **精确相等**，且逐 ID 标签相等；共同缺失、共同新增或共同错标也会失败。图像、文件名和关系从 target 取得，预测行不能覆盖它们。输入顺序不影响输出，内部统一按问题 ID 排序。

训练多数类只使用权威 train targets 的标签序列 `T=(y_1,...,y_N)`，不得接收测试标签来决定类别。若 `sum(T) > N/2` 预测 1，否则预测 0，因此平票确定为 no。

### 3.2 输出

输出 `CLEVRStatistics` 为不可变、只含 Python 标量/tuple 的结构：

- 20 个按 `(method, seed)` 固定顺序排列的 `RunMetrics`；
- 4 个 `SeedSummary`；
- `B-G` 与 `M10-G` 两个 `PairedSummary`；
- 1 个 train-majority `ClassificationMetrics`；
- 1 个 `ClusterBootstrapSummary`，包含 seed、draw 数、quantile 方法、2000 个 replicate 差和区间。

每个分类指标包含问题数、正确数、micro accuracy、可选 cross-entropy、固定关系后固定答案顺序的 8 个 cell，以及可选 eight-cell macro。多数类没有 logits，因此 cross-entropy 为 NA，而不是伪造概率。

## 4. 数学定义

### 4.1 预测、accuracy 与交叉熵

为落实 PUB-R1.1 的 logit 平票规则，

\[
\hat y_q=\mathbf 1[z_{q1}>z_{q0}],
\qquad
\operatorname{Acc}=\frac1Q\sum_{q=1}^{Q}\mathbf 1[\hat y_q=y_q].
\]

严格大于使 `z_{q0}=z_{q1}` 时预测 0。二类平均交叉熵为

\[
\operatorname{CE}=\frac1Q\sum_q
\left(\log(e^{z_{q0}}+e^{z_{q1}})-z_{q,y_q}\right).
\]

实现使用二类稳定式 `numpy.logaddexp(0, z_{1-y}-z_y)`。它在两个极大同值 logits 上仍返回 `log(2)`，避免先加回大平移量再减目标 logit 的抵消；若有限输入的差已超出浮点可表示范围并产生非有限 loss，则硬失败，不静默裁剪 logits。

对每个 `(relation r, label a)` cell，计数 `n_{r,a}`、正确数 `c_{r,a}`、accuracy `c/n`。`n=0` 时 accuracy 为 NA。八格 macro 仅在八格全部非空时定义为八个 cell accuracy 的算术平均；否则 macro 为 NA，不能悄悄改成“非空 cell 平均”。

### 4.2 五 seed 与配对比较

对任一方法的五个 accuracy，报告五个原值、均值、样本 SD 和近似 95% t 区间。对每个 seed：

\[
\Delta_s^{B-G}=A_{B,s}-A_{G,s},\qquad
\Delta_s^{M10-G}=A_{M10,s}-A_{G,s}.
\]

任一五元值 `x_s` 的汇总为

\[
\bar x=\frac15\sum_s x_s,\quad
s_x=\sqrt{\frac{1}{4}\sum_s(x_s-\bar x)^2},\quad
I_t=\bar x\pm 2.7764451051977987\frac{s_x}{\sqrt5}.
\]

这里固定自由度 4；常量完整精度用于计算，并以 NIST 表的三位值 2.776 做独立交叉核对。它是条件于固定数据/协议的小样本近似描述，不是显著性或等效性证明。

### 4.3 图像聚类 bootstrap

令基础测试有 `I>0` 张至少含一个入选问题的图像，按 `(image_index,image_filename)` 排序。每个 replicate 从这 `I` 张图像均匀、有放回抽 `I` 次；其重数向量

\[
W^{(b)}\sim\operatorname{Multinomial}(I;1/I,...,1/I),
\quad b=1,...,2000.
\]

仅创建一个 `Generator(PCG64(31001))` 并一次生成形状 `[2000,I]` 的重数矩阵；全部方法和 seed 共用它。若图像 `i` 有 `n_i` 题，run `(m,s)` 答对 `c_{m,s,i}` 题，则

\[
A_{m,s}^{(b)}=\frac{\sum_i W_i^{(b)}c_{m,s,i}}
                         {\sum_i W_i^{(b)}n_i},\qquad
D^{(b)}=\frac15\sum_s(A_{B,s}^{(b)}-A_{G,s}^{(b)}).
\]

每个重数向量和为 `I`，且每张被抽中的图像的全部问题共同复制。分母按问题数加权，所以不等于先计算每图 accuracy 后等权平均。区间使用 `numpy.quantile(D,(0.025,0.975),method="linear")`；保留全部 2000 个 `D` 支持复算。不得将该区间与训练 seed t 区间合并。

## 5. 手算例与独立数值检查

考虑两张图像：图 A 有 1 题，图 B 有 3 题。某 seed 下 G 分别答对 `(1,0)`，B 分别答对 `(0,3)`。

- 原始 micro：`A_G=1/4`、`A_B=3/4`、差为 `1/2`。
- 若一个 replicate 的重数是 `(2,0)`：两者分母均为 2，G=1、B=0、差=-1。
- 若重数是 `(0,2)`：分母为 6，G=0、B=1、差=1。
- 若重数是 `(1,1)`：分母为 4，恢复原始差 1/2。

如果错误地先平均图像 accuracy，原始 G 与 B 都是 1/2，差为 0；该反例会证伪错误实现。测试必须用字面 expected 值覆盖这三个重数。实际随机函数还需验证形状 `[2000,2]`、每行和为 2、重复调用相同，以及一个独立显式索引/计数实现与生产向量化结果相等。

交叉熵手算：标签 0、logits `(0,0)` 的 loss 为 `log(2)` 且预测 0；标签 1、logits `(-1000,1000)` 的 loss 近似 0。测试使用 `math.log(2)` 与稳定边界，不调用生产 helper 生成 expected。

五 seed 例 `(-0.02,-0.01,0,0.01,0.02)` 的均值为 0，样本 SD 为 `sqrt(0.001/4)=sqrt(0.00025)`，区间对称于 0；expected 由公式字面构造，不复制生产汇总函数。

## 6. 不变量、边界与证伪条件

必须保持：

- target、run 或 row 重排不改变任何指标或 bootstrap replicate；
- 每个 run 的 cohort 和标签精确匹配权威 base-test target，bootstrap denominator 对所有方法/seed 相同；
- 每个 run 的 cell count 总和等于问题数，correct count 总和等于 micro correct；
- accuracy 与 cell accuracy 位于 `[0,1]`；交叉熵非负且有限；
- 同一方法五 seed 汇总严格按 `(101,...,105)`；配对差不能错配 seed；
- bootstrap 每行重数总和为图像数；单图 cohort 的所有 replicate 恢复原指标；
- 函数不修改输入，不读写全局 NumPy RNG；重复调用 bitwise 相同；
- train-majority 的类别只由权威 train targets 决定，平票为 0。

硬失败：空/重复 target、target partition 与官方 split 不符、文件名不匹配、未知/重复/缺失 method-seed、空 run、run 内重复问题 ID、相对权威 target 的 cohort 缺失/新增或标签冲突、非法 relation/label、logits 维度错误或非有限。

以下任一测试结果证伪本实现：手算 micro/CE/cell 不符；重排改变输出；按问题 bootstrap 与生产结果一致而图像手算不一致；共同抽样被每 run 单独推进；五 seed 使用总体 SD；空 cell 被排除后仍给出 macro；tie 预测为 1；缺失或错配 cohort 未失败。

## 7. 成熟库与许可核查

- **NumPy 2.5.3（本地实际版本）**：采用 `numpy.random.Generator(PCG64(seed)).multinomial` 生成重数，采用 `numpy.quantile(..., method="linear")`。官方文档说明 multinomial 输出各类别次数；PCG64 固定 seed 的 bit generator 整数流有兼容保证，但 `Generator` 分布算法的跨版本复现条件更严格，因此最终报告仍必须记录实际 NumPy 版本，不承诺跨版本 bitwise 一致。NumPy 由项目持续维护，BSD-3-Clause。
- **Python 3.13 标准库（本地实际解释器）**：采用 `statistics.stdev`，其官方定义为样本标准差；采用 `math` 做均值区间和稳定 CE。PSF License Version 2。
- **NIST/SEMATECH e-Handbook**：仅作为 `t_{4,0.975}` 三位交叉核对来源（2.776），不复制实现。
- **不采用 SciPy**：当前项目未声明且本地未核实安装；自由度和 seed 数固定，增加依赖无收益。
- **不采用 sklearn/torchmetrics**：通用分类聚合不能自动落实本协议的公共 cohort、八格 NA、配对 seed 和图像聚类抽样；薄的协议层仍不可省。

库文档：

- <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html>
- <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>
- <https://numpy.org/doc/stable/reference/generated/numpy.quantile.html>
- <https://numpy.org/doc/stable/license.html>
- <https://docs.python.org/3/library/statistics.html#statistics.stdev>
- <https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm>

## 8. 后续接口与成本

本单元新增 `src/clevr_statistics.py` 和 `tests/test_clevr_statistics.py`，不修改现有数据/model API。后续报告 PR 将从已发布的 `selected_questions.jsonl` 严格投影权威 `TargetRow`，解析预测 JSONL 为 `PredictionRow`，将统计 dataclass 转成 JSON（`None` 显式写为 `"NA"`），并原子发布完整报告；runner PR 只负责输出逐题 ID、标签和 logits，不得自行提供不可复算的“权威指标”。

时间复杂度为 `O(RQ + BI + BSI)`，其中 `R=20`、`B=2000`、`S=5`；主要额外内存为 `[B,I]` 整数重数矩阵。它不增加模型参数或训练计算。若真实 CLEVR `I` 使矩阵内存不可接受，后续只能用保持同一随机调用顺序和共同抽样语义的分块设计另开修订，不能静默改变算法。
