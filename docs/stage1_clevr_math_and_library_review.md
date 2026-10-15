# Stage 1 CLEVR 数学定义与实现选型

状态：2026-10-15 完成代码前审查；尚未修改 Python 源码，尚未创建实现 PR。

本文只冻结 Stage 1 第一个实现单元所需的数学接口：scene/query schema、单 patch marker 与边界条件。真实数据 loader、训练 runner、采样配方和模型输入改造分别留到后续 PR。

## 1. 坐标域与 patch 映射

设 CLEVR 原图实际宽高为 \(W_0,H_0\)。宽高必须从 PNG 读取并与 scene 记录关联，不从数据集名称或生成器默认值推断。模型输入宽高为 \(W,H\)，patch 宽高为 \(P_x,P_y\)，满足

\[
n_x=W/P_x\in\mathbb N,\qquad n_y=H/P_y\in\mathbb N.
\]

CLEVR `pixel_coords` 的前两项是图像坐标 \((x,y)\)：\(x\) 向右增加，\(y\) 向下增加。有效域定义为半开区间

\[
0\le x<W_0,\qquad 0\le y<H_0.
\]

越界坐标是 schema 错误，必须报告，不能通过 clamp、epsilon 或容差掩盖。对象中心唯一映射到 patch 列、行为

\[
q_x(x)=\left\lfloor\frac{n_xx}{W_0}\right\rfloor,qquad
q_y(y)=\left\lfloor\frac{n_yy}{H_0}\right\rfloor.
\]

因此 \(q_x\in\{0,\ldots,n_x-1\}\)，\(q_y\in\{0,\ldots,n_y-1\}\)。映射直接从原图坐标得到目标 token，不依赖图像 resize 的插值细节。

对 patch \(q=(q_y,q_x)\)，marker channel 在目标输入平面上的定义为

\[
M_q[v,u]=
\begin{cases}
1,&\lfloor u/P_x\rfloor=q_x\ \land\ \lfloor v/P_y\rfloor=q_y,\\
0,&\text{otherwise}.
\end{cases}
\]

也就是整个且仅有一个 patch 的像素为 1。marker 在 resize 后生成，不对 marker 使用双线性或双三次插值。绝对任务得到 `RGB + M_A` 四通道输入；相对任务得到 `RGB + M_A + M_B` 五通道输入。

## 2. 2×2 区域与标签

对 B 的 2×2 区域，patch 的区域索引和局部坐标定义为

\[
r_x(q_x)=\left\lfloor\frac{2q_x}{n_x}\right\rfloor,quad
r_y(q_y)=\left\lfloor\frac{2q_y}{n_y}\right\rfloor,
\]

\[
\ell_x=q_x-r_x\frac{n_x}{2},qquad
\ell_y=q_y-r_y\frac{n_y}{2}.
\]

主协议采用 \(W=H=224\)、\(P_x=P_y=16\)，所以 \(n_x=n_y=14\)，区域边界恰好位于 patch 索引 7。`same_region` 当且仅当 A、B 的 \((r_y,r_x)\) 完全相等；否则为 `cross_region`。

绝对任务标签直接等于 marker 所在 2×2 区域：

\[
y_{abs}=2r_y+r_x\in\{0,1,2,3\}.
\]

相对任务使用有序对象对 \((A,B)\)。令

\[
d_x=x_A-x_B,\qquad d_y=y_A-y_B.
\]

当 \(d_x=0\) 或 \(d_y=0\) 时，四象限方向在对应轴上没有定义，该对象对不产生四分类 query。其余标签由 \((\operatorname{sign}d_x,\operatorname{sign}d_y)\) 唯一确定：左上、右上、左下、右下。这里“上”对应 \(d_y<0\)。不使用 deadband 或数值阈值。

如果 A、B 映射到同一 patch，即 \(q_A=q_B\)，两个 marker 在 token 级不能区分对象身份。该 pair 不属于当前任务的可识别输入域，必须精确排除。这是 patch 映射的单射条件，不是距离阈值。

交换对象次序必须满足：左上与右下互换，右上与左下互换；`same_region` 状态保持不变。

## 3. 与 G/B 相位的关系

G 在一个轴上的相对位置差为 \(q_i-q_j\)。B 的局部平面使用

\[
\ell_i-\ell_j=(q_i-q_j)-(r_i-r_j)\frac n2.
\]

因此：

- 若 \(r_i=r_j\)，A–B 这一位置对的局部相对相位与 G 精确相同；
- 若 \(r_i\ne r_j\)，局部坐标差减去区域原点差；
- 不同区域但局部坐标相同的 patch 在局部平面发生位置别名；
- 局部平面的相位只有在区域原点差乘频率是 \(2\pi\) 的整数倍时才可能偶然与 G 相同，不能笼统声称每个跨区域复相位都不同。

对频率 \(\omega\)，真正应测试的是相位比

\[
e^{\mathrm i\omega\ell_i}\overline{e^{\mathrm i\omega\ell_j}}
=e^{\mathrm i\omega(\ell_i-\ell_j)},
\]

而不是分别比较两个绝对复相位。

上述等价只针对指定 patch 对的局部旋转平面。它不推出 G/B 整层 attention 或最终 logits 在 `same_region` query 上相等，因为场景中的其他 patch 对会参与注意力，且未旋转 CLS 与旋转 patch 之间的相位也会变化。测试不得错误断言整网输出相等。

## 4. schema 的最小字段

第一个实现 PR 的不可变 query record 至少包含：

- `scene_id`、官方 split、`image_filename`；
- 实际 `source_width`、`source_height`；
- `query_kind`：`absolute` 或 `relative`；
- A 的对象索引、原图坐标、patch 坐标、区域坐标；
- relative query 中 B 的同类字段；
- 标签；
- relative query 的 `same_region` 布尔值。

派生字段必须由原始坐标和网格规格计算，不能让调用者同时传入互相可能矛盾的坐标、区域和标签。record 构造失败时返回明确的结构或坐标错误，不静默修正。

## 5. 必须覆盖的数学边界测试

首个 PR 的测试只验证定义，不依赖 CLEVR 文件：

1. 原图四角附近的有效像素分别映射到 patch 网格四角。
2. 2×2 边界前一像素与边界像素分别映射到区域 0 和 1。
3. `x=W_0`、`y=H_0`、负坐标、非正图像尺寸和不可整除的目标网格被拒绝。
4. marker 恰有一个完整 patch 为 1，其余为 0；A/B marker 不共享存储。
5. 四种相对标签均由严格符号得到；坐标轴相等时 query 未定义。
6. 同 patch 的 A/B pair 被拒绝；不同 patch 即使相邻也不因距离被拒绝。
7. 对象交换满足标签对合关系，并保持 same/cross-region 分类。
8. 穷举 14 个轴向 patch 的有序 pair：同区域时局部差等于全局差；跨区域时满足区域原点差公式。
9. 直接与 `src.local_rope.build_axial_cis` 比较相位比，验证同区域局部平面不变量和指定跨区域手算例。

手算边界例：若原图为 480×320、目标网格为 14×14，则 `(239,159)` 映射到 patch `(6,6)`、区域 `(0,0)`；`(240,160)` 映射到 patch `(7,7)`、区域 `(1,1)`。轴向 patch 6 与 7 的 G 差为 \(-1\)，按 A−B 顺序取 `6−7`；B 的局部差为 \(6-0=6\)。patch 2 与 9 的 G 差为 \(-7\)，局部差为 \(2-2=0\)。

独立枚举已覆盖 14×14 的全部轴向有序 pair：98 个同区域 pair 与 98 个跨区域 pair 均满足上述等式；边界像素映射也与手算一致。

## 6. 成熟实现检索与选型

### 官方 CLEVR

- [CLEVR 官方页面](https://cs.stanford.edu/people/jcjohns/clevr/) 是数据格式和许可的权威来源。官方 train/val 提供 scene graph、对象位置和关系；数据许可为 CC BY 4.0。
- [官方生成仓库](https://github.com/facebookresearch/clevr-dataset-gen) 提供 scene JSON 的生成逻辑，软件为 BSD 许可。它在 2023-10-31 被归档，并依赖旧 Blender；只把它作为字段语义的参考，不把生成器引入运行时。
- 官方 `get_camera_coords` 明确给出图像空间 `(px, py)`，其中 `py` 由图像高度减去投影 y 得到，支持本文采用的 y 向下约定。

### PyTorch / torchvision

- 当前环境为 PyTorch 2.6.0、torchvision 0.21.0、timm 1.0.24。
- torchvision 没有官方 CLEVR dataset class；未来 loader 适合薄封装标准 [`VisionDataset`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VisionDataset.html)，图像解码使用 PIL，张量和模型操作继续使用 PyTorch。
- patch projection 已由上游 RoPE-ViT 通过 timm `PatchEmbed` 实现，并原生支持 `in_chans`。后续模型 PR 应传入 4/5，而不是重写 patchify。

### 未采用的候选

- Meta VISSL 有 CLEVR count/distance 的数据准备脚本，但项目也已归档，输出任务与本研究的对象对 marker/schema 不同，不复用其任务构造。
- Hugging Face Datasets 能读取通用 JSON，但没有为本任务提供官方、类型化的对象对接口；引入 Arrow/cache 层不会减少关键派生逻辑，还可能触发下载或完整性检查。Stage 1 使用用户提供的本地官方文件，不采用该依赖。
- Pydantic/jsonschema 可表达通用字段约束，但无法替代 patch、region、label 之间的数学一致性。首个 PR 优先使用标准库不可变 record 与单一构造函数，避免增加只用于重复校验的运行依赖。

结论：复用官方 CLEVR 字段定义、torchvision `VisionDataset` 接口、PIL 解码、PyTorch 张量操作和上游 timm `PatchEmbed`；只自行实现本研究特有的 query 派生和 marker 构造。该自定义部分保持最小，并由第 5 节不变量约束。

## 7. 证伪条件与实现边界

以下任一情况会否定当前定义或阻止进入训练实现：

- 官方 scene 的坐标语义与图像实际尺寸不满足本文有效域；
- marker 经过 patch projection 后激活多个 token，或 A/B 的 query 身份在允许域内不可区分；
- same-region 相位比不能与 G 对齐；
- G/B 为支持 4/5 通道而获得不同参数量或不同非 PE 初始化；
- 数据构造需要靠距离阈值、clamp 或手工修补才能产生一致标签。

通过这些结构检查只说明实现符合数学定义，不说明 B 会提高准确率。性能主张仍必须由后续预注册的配对实验决定。
