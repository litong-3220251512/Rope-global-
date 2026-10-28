# CLEVR PUB-R1.1 预测解析与统计报告设计

**状态：** 已冻结供实现；独立 builder review 的四项 Important 与两项 Minor 均已纳入；用户已授权按已验收的 PUB-R1.1 持续推进，不逐单元等待确认。
**协议权威：** `docs/rope_public_data_experiment_plan.md`（PUB-R1.1）第 7、10 节。
**上游接口：** `src/clevr_statistics.py` 的不可变目标、预测与统计对象。

## 1. 意图、成功条件与边界

统计核心已经能从内存对象独立复算正式网格，但真实实验还需要一个文件信任边界。本单元负责：

- 从用户显式指定的完整审计 artifact 目录读取固定文件名，交叉核对协议版本、图像分区、partition 汇总与 `selected_questions.jsonl`，并只把 `train` 与 `base_test` 投影成权威 `TargetRow`；
- 从用户逐个显式指定的 20 个自描述预测 JSONL 解析 `G/B/M10/Q-only × 101..105`，逐行核对协议/method/seed，不扫描目录、不猜文件名、不接受 runner 汇总指标；
- 调用统计核心，生成包含全部 run、五 seed 汇总、配对差、train-majority 与 2000 个图像聚类 bootstrap replicate 的确定性报告；
- 在新目录中共同发布 `statistics_report.json`、规范化目标 JSONL 和全部 20 份规范化预测 JSONL，使报告包本身可独立复算；同时记录输入的直接可解释证据：解析路径、字节数、行数、方法、seed 与环境版本，不计算摘要或哈希。

非目标：生成预测、读取 RGB/scene/checkpoint、判断 checkpoint 是否为第 8000 步、验证训练完成状态、记录 optimizer/sampler/RNG、启动训练或评估、监控进程、作性能或显著性结论。runner 的运行状态与 checkpoint provenance 属于后续独立单元；本报告只证明“这些明确给出的逐题文件在权威 cohort 上复算为何值”。

反目标：自动发现“看起来像”的预测文件、容忍缺失 run、接受最好 seed、读取 runner 提供的 accuracy、从测试标签选择多数类、把 `null` 当作 NA 输入、静默接受重复 JSON 键、NaN/Infinity、空行、额外字段或旧 schema。

## 2. 已考虑的实现路径

1. **显式重复 `--prediction-file PATH`，文件逐行自描述（采用）。** 20 个路径都由调用者指定；每行携带协议、method 与 seed，解析器要求文件内一致并形成精确 formal grid。它最符合“不自动搜索替代数据”，又不让调用者给同一内容任意贴 G/B 标签。代价是每行重复少量元数据和命令较长，可由后续 runner 生成命令但不能绕过同一 API。
2. **只给 `--predictions-dir` 并按约定文件名扫描。** 命令较短，但引入自动发现、额外文件处理和命名约定；目录里出现旧 run 时容易误用，不采用。
3. **新增 run-index JSON 再由它引用 20 个文件。** 对完整 runner provenance 最终可能有用，但本单元还没有已冻结的 checkpoint/训练状态 schema；现在引入会把未定义状态伪装成可信元数据，不采用。逐行 method/seed 只绑定预测 artifact 的声明身份，不能证明模型确实按该方法训练；后续 runner contract 负责生成与运行记录的一致声明。

输出采用一个全新目录而不是覆盖单文件：在目标同一父目录创建私有 staging 目录，写完后 rename 为最终目录。这样协作读者只会看见“无目录”或“含完整 bundle 的目录”，且沿用数据审计 CLI 已验证的发布习惯。它不宣称断电持久性。

## 3. 输入 schema 与类型

### 3.1 审计 artifact 目录

CLI 接收显式 `--audit-dir`，且只读取其中固定的 `image_partitions.json`、`selected_questions.jsonl` 与 `audit_report.json`；不递归扫描、不寻找替代文件。`vocabulary.json` 不参与结果统计，报告器不读取它。所有 JSON 使用第 3.4 节的严格解析。

`image_partitions.json` 顶层字段必须精确为 `protocol_version, partitions`，协议必须为 `PUB-R1.1`。`partitions` 必须精确包含 `train,dev,base_test,confirmation`；每个元素必须精确含 `official_split,image_index,image_filename`，并能构造 `ImageIdentity`。每张图像全目录唯一，train 与 dev 只含 official train，base_test 与 confirmation 只含 official val。

`audit_report.json` 顶层必须精确包含审计发布器现有的 `protocol_version,environment,partition_summaries,cell_counts,exclusion_counts,source_summaries,file_summaries`。协议必须为 `PUB-R1.1`。报告器只信任并严格解析与本单元有关的 `partition_summaries`：四个元素按固定 partition 顺序，每项精确含 `partition,mother_image_count,selected_image_count,selected_question_count` 和非负严格整数；其余五个字段必须具有审计发布器产生的 JSON container 类型，但不重新执行源 CLEVR 审计。

解析 `selected_questions.jsonl` 后，逐行图像必须出现在 `image_partitions.json` 的同名 partition；每个 partition 的唯一入选图像数与问题行数必须分别等于对应 summary 的 `selected_image_count`、`selected_question_count`，summary 的 `mother_image_count` 必须等于 image partition 长度。由此建立“统计条件于这个已发布 PUB-R1.1 审计 bundle”的信任边界；它仍不能证明 bundle 从官方原始数据正确生成，原始数据审计是上游职责。

### 3.2 审计 manifest

审计 bundle loader 逐行读取 `selected_questions.jsonl` 的 UTF-8 JSON。空文件、空行、非对象、重复键、非标准常量、缺失/额外字段均硬失败。每行字段集合必须精确为：

```text
partition, official_split, image_index, image_filename, image_path,
question_index, source_position, question, encoded_tokens, answer, relation
```

字段约束：

- `partition ∈ {train,dev,base_test,confirmation}`；映射必须为 `train→train`、`dev→train`、`base_test→val`、`confirmation→val`；
- `image_index, question_index, source_position` 是非负 Python `int` 对应的 JSON integer，布尔值不接受；
- `image_filename = CLEVR_<official_split>_<image_index:06d>.png`；
- `image_path = images/<official_split>/<image_filename>`，必须是上述规范 POSIX 相对路径；不访问该图像；
- `question` 是字符串；`encoded_tokens` 是非负整数数组；报告器不重新定义上游文本内容约束；
- `answer ∈ {0,1}`，`relation ∈ {left,right,front,behind}`。

所有 partition 都先做 schema 验证。`(official_split, question_index)` 与 `(official_split, source_position)` 分别唯一；每个图像身份必须与第 3.1 节的分区记录完全匹配。随后仅将 `train` 和 `base_test` 转为：

\[
t=(partition, official\_split, image\_index, image\_filename,
question\_index, relation, label=answer).
\]

目标按固定 partition 顺序 `train,base_test`、再按 `question_index` 规范排序，并继续由统计核心要求两者均非空。dev 与 confirmation 不参与任何指标或多数类选择。

### 3.3 预测 JSONL

一个 `PredictionSource(path)` 只包含显式文件路径。20 个 resolved path 必须互不相同，避免同一 artifact 被意外复用。每个文件逐行对象字段必须精确为：

```json
{"protocol_version":"PUB-R1.1","method":"B","seed":101,
 "question_index":123,"label":1,"logits":[-0.25,0.75]}
```

协议必须逐行等于 `PUB-R1.1`；method/seed 必须逐行相同并属于 formal set。它们是该文件声明的运行身份，不是训练过程真实性证明。`question_index`、`label` 和两个有限 real logit 的完整约束交给 `PredictionRow` 再验证；输入 JSON array 转成不可变 tuple。文件为空、含空行、重复 question ID、字段缺失/额外、重复键、NaN/Infinity 或非对象均失败。文件内顺序不影响结果。

CLI 必须先验证正好有 20 个互异 resolved path；读取其自声明身份后，20 个 source 必须构成精确 Cartesian product：

\[
\mathcal R=\{G,B,M10,Q\text{-}only\}\times\{101,102,103,104,105\}.
\]

重复、缺失或额外 key 均失败；规范输出顺序为 method 后 seed。随后统计核心再次逐 run 检查问题 ID 集合与权威 base-test 完全相等、复制标签一致。无法从 artifact 内部证明“B 文件确由 B 模型生成”；允许表述仅为“文件声明为 B 且其逐题内容通过协议复算”。

### 3.4 严格 JSON 语义

Python `json` 默认会保留重复对象键中的最后一个，也默认接受 NaN/Infinity。解析器使用 `object_pairs_hook` 检测任意层级重复键，并用 `parse_constant` 拒绝非标准数值；所有错误都附带文件路径和 1-based 行号。输出用 `allow_nan=False`。

本地输入可能较大，但不设任意字节阈值：manifest 与 prediction 的每行 schema 都是小对象，逐行读取使常驻解析内存为单行加最终不可变 rows；是否限制不可信外部输入大小不属于本地实验 artifact 契约。

## 4. 输出对象与 JSON

纯 API：

```text
PredictionSource(path: Path)
ArtifactEvidence(path: str, size_bytes: int, record_count: int)
PredictionEvidence(method, seed, source_path, size_bytes, row_count,
                   canonical_path)
PartitionCounts(train, dev, base_test, confirmation)
EnvironmentEvidence(python, numpy)
StatisticsReport(protocol_version, report_schema_version, audit_sources,
                 targets_path, manifest_partition_counts, predictions,
                 statistics, environment)
StatisticsBundle(report, targets, runs)

load_audit_targets(audit_dir) ->
    (targets, tuple[ArtifactEvidence, ...], PartitionCounts)
load_prediction_run(source) -> (RunPredictions, PredictionEvidence)
build_statistics_bundle(audit_dir, prediction_sources) -> StatisticsBundle
statistics_report_payload(report) -> dict[str, object]
publish_statistics_bundle(output_dir, bundle) -> None
```

`StatisticsBundle` 是发布前的内存对象：`targets` 与 `runs` 只用于写规范化 JSONL，`statistics_report_payload` 不把它们再次内嵌进 report JSON。全部记录保持不可变 tuple，并以统计核心的规范顺序存放。

路径在 evidence 中记录为 `expanduser().absolute().resolve()` 后的 POSIX 字符串。每个输入通过已打开文件描述符的 `fstat` 记录实际读取文件的 `st_size`。audit evidence 的 `record_count` 对 image partition 是四个列表的图像总数、对 selected manifest 是 JSONL 行数、对 audit report 是 1；prediction evidence 的 `row_count` 是 JSONL 行数。这里的 path/size/count/schema 是直接证据，不声称内容身份验证。

发布包固定包含：

- `statistics_report.json`；
- `targets.jsonl`：按 partition 后 question ID 排序的 train/base-test `TargetRow` 七字段；
- `predictions/` 下 20 个固定名 `<method>_seed_<seed>.jsonl`，其中 `Q-only` 原样保留；每行使用第 3.3 节六字段并按 question ID 排序。

规范化文件是解析后值的重新编码，不是 source 的字节复制；它们与 report 在同一个 staging 中产生并共同提交。由此报告包单独保留独立复算所需的目标与全部逐题 logits。它不是防篡改容器：后续读取者应重新解析规范化文件并复算，与 report 对照，而不能仅凭 path/size/count 假定内容未变。

报告 JSON 顶层固定包含：

- `protocol_version: "PUB-R1.1"` 与 `report_schema_version: 1`；
- `environment`：实际 Python 与 NumPy 版本；
- `audit_sources`：按 `image_partitions.json,selected_questions.jsonl,audit_report.json` 顺序的 source evidence；`targets_path` 固定为规范目标相对路径 `targets.jsonl`；
- `manifest_partition_counts`：精确四个 partition key；
- 20 个规范排序的 prediction evidence，每项另含对应规范预测相对路径；
- `statistics`：统计 dataclass 的完整递归序列化。

tuple 写为 JSON array，dataclass 按声明字段写为对象。仅三个数学可空位置允许转换为 `"NA"`：`CellMetrics.accuracy`、`ClassificationMetrics.macro_accuracy`、`ClassificationMetrics.cross_entropy`；其他位置出现 `None` 必须失败。普通字符串不得被解释为 NA 输入。输出保留全部 2000 个 bootstrap replicate，以支持直接复算 quantile。报告不含时间戳、随机 UUID 或由 runner 声称的指标，因此同内容、同 resolved source 路径、同版本环境下字节确定。

全部 JSON/JSONL writer 固定 `ensure_ascii=False, sort_keys=True, allow_nan=False`；完整 JSON 另固定 `indent=2`，JSONL 使用默认带空格 separators，每条记录及文件末尾各一个 `\n`。`statistics_report.json` 的精确顶层 key 为 `protocol_version,report_schema_version,environment,audit_sources,targets_path,manifest_partition_counts,predictions,statistics`；`report_schema_version` 是整数 `1`，environment 精确为 `{"python": <platform.python_version()>, "numpy": <importlib.metadata.version("numpy")>}`。

## 5. 原子发布与失败语义

CLI：

```text
python run_clevr_statistics.py \
  --audit-dir /explicit/audit-output \
  --prediction-file /explicit/G-101.jsonl ... \
  --output-dir /new/statistics-output
```

`--prediction-file` 必须正好出现 20 次并经纯 API 验证；`argparse` 只负责语法，不替代 domain validation。

发布过程：

1. 目标解析为绝对路径；若已存在（即使为空）或是 symlink，拒绝覆盖。
2. 创建父目录和 `.<name>.lock`，使用 `O_CREAT|O_EXCL` 防止并发发布。
3. 用 `tempfile.mkdtemp(dir=output.parent)` 在同一父目录创建 staging。
4. 以第 4 节固定编码写规范 targets、20 份规范 predictions 和 `statistics_report.json`。
5. 再检查目标未出现，记录 staging directory 的 `(st_dev, st_ino)`，然后把 staging directory rename 为目标。
6. rename 是提交点。若 rename 调用在完成文件系统移动后抛出异常，只有旧 staging 路径已不存在且最终路径的 `(st_dev, st_ino)` 与记录值相同时才按已提交处理：清除 staging 清理所有权、保留完整最终目录，并进入提交后的 lock 处理；否则仍按提交前失败处理。提交前任意主异常先清理本次 staging，再尝试清理 lock，且不创建最终目录；提交后不回滚最终目录。lock unlink 自身失败时抛出专用状态错误：rename 前明确 `not committed`，rename 后明确 `committed`；两者都保留无法删除的 lock 供人工审查。

成功 rename 在项目支持的 POSIX 环境中提供原子可见性；跨文件系统风险由同父目录 staging 排除。no-overwrite 只对遵守同一 `O_EXCL` lock 的协作发布者成立；非协作进程在最终检查与 rename 之间创建空目标目录的 POSIX 竞态不在保证内，设计不以 ctypes 私自引入 Linux-only `renameat2`。没有 fsync，因此不承诺系统掉电后的持久性。进程被 SIGKILL 时可能留下 lock/staging，工具不会猜测其是否 stale 或自动删除；由操作者审查后处理。

## 6. 不变量、边界与证伪条件

必须保持：

- manifest 行、source 参数和预测行重排不改变 `statistics` 或规范化 artifact 字节；evidence 按固定 grid 输出；
- dev/confirmation 标签变化不能改变 train-majority 或任何正式指标，只能改变 manifest partition counts（若行数改变）；
- runner 写入的任何汇总字段都因额外字段被拒绝，不能进入统计；
- 每个 prediction 文件行数等于权威 base-test target 数，且 20 个来源都经统计核心做 ID/label 精确匹配；同一 resolved 文件不能重复占多个 run；
- JSON 中 `true` 不可冒充 1，重复键不可被 last-write-wins，NaN/Infinity 不可进入计算；
- `"NA"` 只在输出中表示数学未定义值，输入不接受它替代数值或 label；
- 输入、序列化或 rename 的提交前失败会清理本次 staging；在 lock unlink 正常时也清理 lock，且无最终目录；协作发布者的已有目标永不覆盖；lock unlink 自身失败时保留 lock 并明确报错；
- 不修改输入、不读取 RGB/scene/checkpoint、不访问网络、不改变 NumPy 全局 RNG。

边界：路径记录依赖本机位置；不用于内容等同性。一个文件在读取期间被外部原位改写不属于支持的并发输入；正式 workflow 应先完成 immutable artifact 发布，再运行报告器。报告器只能验证文件与协议 schema/数学的一致性，不能证明训练确实按协议执行。

以下任一结果证伪实现：重复 JSON 键被接受；同一 resolved path 被复用；文件内 method/seed 不一致；缺一个 run 仍出报告；改变 dev 标签改变统计；额外 runner accuracy 覆盖复算值；prediction ID/label 错配未失败；数学可空字段以外的 `None` 被转换；普通提交前失败后出现 final/staging 或可正常删除却残留的 lock；输出遗漏规范 targets、任一 prediction 或 bootstrap replicate；输入顺序改变统计。

## 7. 手算与独立检查

最小 manifest 含 train 标签 `(0,1)` 和八个 base-test 关系×答案 cell。train 平票必须预测 no；20 个 prediction 文件都用 `(0,0)` logits 时，所有 run micro accuracy 为 `4/8=0.5`、CE 为 `log(2)`、no cells 为 1、yes cells 为 0、macro 为 0.5，B−G 与 M10−G 的五 seed 值及 bootstrap replicates全为 0。测试 expected 直接写这些值，不调用 serializer 或统计 helper 生成。

将一行写成 `{"label":0,"label":1,...}` 必须在构造 `PredictionRow` 前失败；将 logits 写成 `[NaN,0]` 必须由 strict decoder 失败。少给或多给一个 source、或重复同一 resolved path，必须在打开任何 prediction path 前失败；读完 20 个文件后再以自声明身份检查 formal grid。

原子性测试分别在 JSONL/JSON 写入中途和 rename 时注入异常，直接检查 final、glob 匹配的 staging 与 lock 均不存在；已有目标文件/目录保持原内容不变。另在 rename 已成功后注入 lock unlink 失败，断言专用错误明确写“committed”，final 与 lock 保留且 staging 已消失；在提交前注入同一故障则错误明确写“not committed”，final/staging 不存在而 lock 保留。

## 8. 成熟库与许可核查

- **Python 3.13.11 标准库（采用，PSF License 2.0）**：`json` 提供 `object_pairs_hook` 检测重复键、`parse_constant` 拒绝 NaN/Infinity，且 `allow_nan=False` 可强制标准输出；`argparse` 是标准基本 CLI 解析器；`tempfile.mkdtemp` 在指定父目录安全创建 staging；`os.open(...O_EXCL...)` 与同文件系统 rename 支持本项目限定于协作发布者的发布语义。
- **NumPy 2.5.3（已有依赖，BSD-3-Clause）**：仅由上游统计核心使用并在报告 environment 中记录；本单元不新增数值实现。
- **不采用 jsonschema/Pydantic**：项目未声明这些依赖；schema 很小且还需构造严格领域 dataclass、定位 JSONL 行号与拒绝重复键，新增依赖不能替代这些任务特有检查。
- **不采用 pandas/Polars**：逐行小对象与精确错误语义不需要 dataframe，且不能替代 formal-grid 和权威 cohort 验证。
- **不采用第三方 atomicwrites**：发布的是新目录而非覆盖文件；标准库同父目录 staging + rename 已覆盖所需原子可见性，避免新增依赖。

官方文档：

- <https://docs.python.org/3.13/library/json.html>
- <https://docs.python.org/3.13/library/argparse.html>
- <https://docs.python.org/3.13/library/tempfile.html#tempfile.mkdtemp>
- <https://docs.python.org/3.13/library/os.html#os.rename>
- <https://numpy.org/doc/stable/license.html>

## 9. 文件、测试与后续接口

本单元新增 `src/clevr_reporting.py`、`run_clevr_statistics.py`、`tests/test_clevr_reporting.py`，不修改统计公式、模型、dataset 或已有数据审计 CLI。实现按三层测试驱动：严格 audit-bundle/prediction 解析；完整 report/payload 与规范化复算 artifact；CLI 与发布故障。

后续 runner 必须输出这里定义的最小逐题 schema，但其 checkpoint、状态、日志、配置与失败 run artifact 另行冻结。它可以生成 20 个显式 CLI 参数，不能把自己计算的 accuracy 注入本报告。待用户提供真实预测并明确授权基础测试结果分析后，代理才运行此入口、读取报告并独立解释；没有真实输入时，本单元只有工程证据。
