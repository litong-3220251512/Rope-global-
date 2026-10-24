# CLEVR PUB-R1.1 checkpoint 与精确恢复设计

**状态：** 待独立 builder review；用户已授权按已验收的 PUB-R1.1 连续推进，不逐单元等待确认。
**协议权威：** `docs/rope_public_data_experiment_plan.md`（PUB-R1.1）第 5.2、5.4、6、10 节。
**分支边界：** 本单元堆叠在 `feat/clevr-paired-sampler`；只依赖真实 `CLEVRIndexStream` 和 PyTorch 通用状态接口，不合并尚未落地的模型、数据、训练配置或统计分支。

## 1. 意图、成功条件与非目标

训练恢复不能只加载模型参数。一次完整更新边界 (k) 的可恢复状态定义为

\[
S_k=(\theta_k, O_k, L_k, I_k, R_k, M_k),
\]

其中：

- \(\theta_k\)：模型全部参数与持久 buffer 的 `state_dict`；
- \(O_k\)：AdamW 参数组与逐参数状态；
- \(L_k\)：`LambdaLR` 状态；
- \(I_k\)：`CLEVRIndexStream` 当前置换、已消费游标与独立 PCG64 状态；
- \(R_k\)：Python、NumPy、PyTorch CPU，以及调用者明确列出的 CUDA device 的 RNG 状态；
- \(M_k\)：协议、schema、method、run seed、完成更新数、数据集大小和创建环境。

成功条件是把一个新进程中已按相同配置构造的对象恢复为 \(S_k\)，并直接证明：

1. 模型、optimizer、scheduler、sampler 和 RNG 状态逐项相等；张量用 `torch.equal`，不用哈希；
2. `sampler.preview(128)` 与未中断运行的下一有效 batch 样本索引相同；
3. Python、NumPy、PyTorch CPU（以及可用时的指定 CUDA device）下一随机抽样相同；
4. checkpoint 只代表完整 optimizer 更新边界，不保存或恢复半个梯度累积窗口；
5. 文件发布后读者只会看到“不存在”或完整的单文件 checkpoint；已存在目标不被覆盖。

本单元不是训练 runner。它不读取 RGB、问题、scene、audit bundle 或模型权重，不创建 DataLoader，不执行 forward/backward/optimizer update，不决定 pilot/正式 checkpoint 文件名，不记录逐更新 loss、逐题预测或完成/失败日志，也不启动工程冒烟、pilot、正式训练或评估。runner 后续负责把本模块与真实 QA 模型、数据 batch、固定学习率公式和运行目录连接起来。

## 2. 方案比较与选择

### 2.1 跨分支组织

1. **在 sampler 分支上实现模型无关、协议专用的 checkpoint 核心（采用）。** 使用真实 `CLEVRIndexStream`，模型只要求 `nn.Module`，优化器/调度器严格要求 AdamW/LambdaLR。这样可以验证最容易被 DataLoader 预取混淆的“已消费游标”，又不把尚未合并的模型、数据、训练配置 PR 重复塞入本 PR。代价是学习率公式与真实 QA 参数组只能在后续 runner 集成测试中再交叉核对。
2. **等待全部依赖 PR 合并后一次实现。** 集成上下文最完整，但会让独立 checkpoint 契约停滞，且用户已要求 PR 阻塞时继续完成不依赖它的工作，不采用。
3. **把 model/data/sampler/training-config 分支合并成临时大分支。** 可立即做端到端测试，但 PR 会重复多个已审查功能并破坏“一 PR 一问题”，不采用。

### 2.2 序列化与发布

1. **`state_dict` + `torch.save`，`torch.load(..., map_location="cpu", weights_only=True)`（采用）。** PyTorch 2.6 官方 serialization note 推荐保存 module `state_dict`，并说明 `weights_only=True` 仅允许张量、基本类型和必要容器且不动态 import。payload 因此只含张量、字符串、普通数值、tuple/list/dict/`OrderedDict`；不保存 module、函数或自定义类实例。[PyTorch 2.6 serialization semantics, “Saving and loading torch.nn.Modules” 与 “torch.load with weights_only=True”](https://docs.pytorch.org/docs/2.6/notes/serialization.html)
2. **JSON + 多个 tensor 文件。** 人类可读，但 model/optimizer/RNG 会拆成多文件事务，增加不一致窗口，不采用。
3. **safetensors。** 适合纯 tensor 权重，但 optimizer、scheduler、sampler 和四类 RNG 仍需第二种 schema，并新增依赖；不能单独满足本契约，不采用。
4. **任意 pickle / `weights_only=False`。** 会扩大反序列化执行面，且没有必要，不采用。

单文件先写入目标同一父目录中的私有 staging 文件，关闭后以 `os.link(staging, target)` 发布。硬链接创建在目标存在时原子失败，因而不覆盖并发出现的目标；成功 link 是 commit point。随后 unlink staging。若 link 完成其文件系统副作用后异步异常，比较 staging 与 target 的 `(st_dev, st_ino)`：只有 target 是同一 inode 才归类为已提交。该设计是 POSIX 本地文件系统契约，不承诺 `fsync` 断电持久性、SIGKILL 清理、网络文件系统语义或恶意目录替换。[Python 3.11 `os.link`](https://docs.python.org/3.11/library/os.html#os.link)、[`tempfile.mkstemp`](https://docs.python.org/3.11/library/tempfile.html#tempfile.mkstemp)

## 3. API 与固定 schema

新增 `src/clevr_checkpoint.py`：

```python
class CLEVRCheckpointError(ValueError): ...

class CLEVRCheckpointPublicationError(CLEVRCheckpointError):
    committed: bool

@dataclass(frozen=True, slots=True)
class CLEVRCheckpointMetadata:
    path: str
    size_bytes: int
    method: str
    run_seed: int
    completed_updates: int
    dataset_size: int
    cuda_devices: tuple[int, ...]

def save_clevr_checkpoint(
    path: Path,
    *,
    method: str,
    run_seed: int,
    completed_updates: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> CLEVRCheckpointMetadata: ...

def load_clevr_checkpoint(
    path: Path,
    *,
    expected_method: str,
    expected_run_seed: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    sampler: CLEVRIndexStream,
    cuda_devices: tuple[int, ...],
) -> CLEVRCheckpointMetadata: ...
```

`method` 精确属于 `G/B/M10/Q-only`；`run_seed` 是非负普通 Python `int`；`0 <= completed_updates <= 8000`；`cuda_devices` 是严格递增、无重复、非负普通整数 tuple。空 tuple 是 CPU 测试/运行契约，不探测或初始化 CUDA。非空 tuple 只允许在 CUDA 已由调用者初始化后使用，模块不替 runner 选择设备。

目标 payload 顶层字段精确为：

```text
protocol_version, checkpoint_schema_version,
method, run_seed, completed_updates, dataset_size,
model_state, optimizer_state, scheduler_state, sampler_state,
rng_state, environment
```

固定值为 `protocol_version="PUB-R1.1"`、`checkpoint_schema_version=1`。`environment` 精确记录 `python,numpy,torch,cuda_runtime` 字符串；load 要求与当前进程完全一致。该严格性用于限定“同锁定软件栈精确恢复”，不宣称跨版本或跨硬件 bitwise 一致。

`rng_state` 精确包含：

```text
python, numpy, torch_cpu, cuda_devices, torch_cuda
```

- Python 使用 `random.getstate()` / `setstate()`；官方文档明确 state 可用于恢复生成器到获取时的位置。[Python 3.11 random bookkeeping](https://docs.python.org/3.11/library/random.html#bookkeeping-functions)
- NumPy 全局 legacy RNG 使用 `numpy.random.get_state(legacy=True)`；返回 `MT19937` 名称、624 个 uint32 key、位置、cached-Gaussian 标志和值。为保持 `weights_only=True`，key 转成 CPU `torch.int64`，load 时检查 `[0,2^32)` 后转回 `numpy.uint32`。[NumPy 2.2 `numpy.random.get_state`](https://numpy.org/doc/2.2/reference/random/generated/numpy.random.get_state.html)
- PyTorch CPU 使用 `torch.random.get_rng_state()` 的 CPU ByteTensor；官方文档明确它只对应默认 CPU generator。[PyTorch 2.6 `torch.random`](https://docs.pytorch.org/docs/2.6/random.html)
- 每个显式 CUDA device 分别保存 `torch.cuda.get_rng_state(device)` 返回的 CPU ByteTensor，并用对应 `set_rng_state` 恢复；不调用会隐式枚举/初始化所有设备的接口。无 GPU 的当前环境只验证空 tuple 路径，不虚构 CUDA 通过证据。

所有 state 在写入 payload 前深拷贝；张量 `detach().cpu().clone()`，避免后续训练原位修改已捕获对象。`CLEVRCheckpointMetadata.path` 使用最终文件的绝对 resolved 路径，`size_bytes` 来自已发布文件 descriptor/`stat`，不计算 SHA、MD5 或任何自定义摘要。

## 4. 完整更新边界与 sampler 数学契约

设有效 batch 大小固定为 \(B=128\)，训练集大小为 \(D>0\)。`CLEVRIndexStream.state_dict()` 中当前 epoch 为 \(e\)，游标为 \(c\in[0,D]\)。已消费样本数必须满足

\[
C(I_k)=eD+c=Bk=128k.
\]

该等式是 checkpoint 是否处在完整 optimizer 更新边界的直接判据。它使用 `commit` 后的游标，而不是 DataLoader 已预取但尚未完成更新的索引。保存前必须同时满足：

- `sampler.run_seed == run_seed`；
- `sampler.dataset_size == dataset_size`；
- 上式成立；
- 模型所有 parameter 的 `.grad is None`，表示调用者已经在 optimizer/scheduler step 后执行 `zero_grad(set_to_none=True)`，不存在待恢复的半窗口梯度。

边界 `c=D` 合法。例如用缩小的手算示例 \(D=5,B=4\)（仅说明不变量；正式协议仍为 128），seed101 的首段流为 `(0,2,1,4,3,4,3,1,...)`。完成一次更新后 `e=0,c=4`，所以 `eD+c=4=Bk`，下一 batch 从 `(3,4,3,1)` 开始；恢复不得重新从 `(0,2,1,4)` 开始。若正好消费 5 个样本，则 `e=0,c=5` 仍表示消费 5；下一次 preview 才进入下一置换。

可证伪条件：任一 checkpoint 的 `eD+c != 128k`、恢复后下一 128 个索引不同、预取改变已保存游标、或恢复从整轮开头重新取样，都推翻实现正确性。

## 5. scheduler、optimizer 与模型状态约束

本单元接受已经构造并相互连接的 `nn.Module`、`AdamW` 和 `LambdaLR`。`scheduler.optimizer is optimizer` 必须成立。PyTorch 文档提醒恢复时先构造 scheduler，再加载 optimizer state，以免构造 scheduler 覆盖已加载 learning rate；本 API 要求对象已经全部构造，并按 `scheduler.load_state_dict` 后 `optimizer.load_state_dict` 的顺序应用状态。[PyTorch `Optimizer.load_state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html)、[`LRScheduler`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html)

保存与加载前检查：

- model state key、shape、dtype 与目标模型完全一致；
- optimizer state 顶层结构、参数组数量、每组 `group_name`、`param_names`、参数顺序与当前 optimizer 一致；不能靠位置把另一个架构或方法的状态贴到当前参数；
- scheduler 的 `last_epoch == completed_updates`、`_step_count == completed_updates + 1`，`_last_lr` 与 optimizer 每组当前 `lr` 逐项相等；
- checkpoint 内 sampler state 先由一个同 `dataset_size/run_seed` 的临时 `CLEVRIndexStream` 完整验证，再碰 live sampler。

本分支没有 `build_clevr_adamw`/`build_clevr_lr_scheduler`，因此本模块不重复实现或声称验证 0.05 参数分组、betas/epsilon 或 \(\eta_k\) 公式。后续 runner 集成必须用现有 training-config 工厂构造对象，并增加 checkpoint 元数据、optimizer 参数名、`last_epoch` 和下一学习率的交叉回归。

## 6. 加载事务与错误语义

`load_clevr_checkpoint` 先以 `map_location="cpu", weights_only=True` 完整读取，再验证 exact schema、普通整数（拒绝 bool/NumPy scalar）、环境、run identity、所有 state 结构和 sampler/完整更新不变量；验证失败时 live 对象和全局 RNG 不得变化。

目标 model 的全部 parameter 在 load 入口也必须满足 `.grad is None`。恢复契约不携带梯度，因此带有半个累积窗口的 live 对象不能被静默覆盖成“完整更新边界”。

进入应用阶段前，函数深拷贝 live model/optimizer/scheduler/sampler 与全部指定 RNG。应用顺序为 model → scheduler → optimizer → sampler → Python/NumPy/PyTorch CPU/CUDA RNG。任一 `BaseException` 触发反向恢复；恢复后逐项直接比较。若主错误与 rollback 错误同时发生，抛出 `CLEVRCheckpointError`，保留主错误为 cause，并把 rollback 错误附在异常 note；不得声称状态完整恢复。成功后也逐项与 checkpoint 直接比较，再返回 metadata。

这提供进程内“成功全部恢复，失败保持进入函数前状态”的事务目标，但不能对进程被 SIGKILL、CUDA driver 崩溃或硬件错误作保证。故障注入测试必须覆盖 model/optimizer/scheduler/sampler/RNG 应用失败与 rollback；只断言 mock 调用次数不算证据。

## 7. 文件事务与故障边界

保存步骤：

1. 严格验证 path 为 `Path`，父目录必须已经存在且为目录，目标必须不存在（包括 symlink）；
2. 在同一父目录 `mkstemp(prefix=f".{name}.staging-", mode=0o600)`；
3. 对打开的二进制 descriptor 写 `torch.save(payload, file)`，flush、关闭；不做 hash；
4. 捕获 staging `(st_dev, st_ino)`，调用 `os.link(staging, target)`；成功 link 是 commit point；
5. unlink staging；返回 final metadata。

所有 post-staging `BaseException` 都进入清理。commit 前错误只删除本 invocation 的 staging；commit 后绝不删除 final。若异常发生在 link 返回边界，只有 target 与 staging identity 都证明同一 inode 才升级为 committed。清理失败抛 `CLEVRCheckpointPublicationError(committed=...)`，消息明确 `committed` 或 `not committed`；原始错误和清理错误都保留。测试用真实临时目录断言最终文件、staging 和既有目标状态。

不保证：断电持久性（无 `fsync`）、SIGKILL 后 staging 自动回收、恶意父目录替换、网络文件系统 hard-link 语义、任意大恶意 checkpoint 的资源上限。`weights_only=True` 缩小任意代码执行面，但 PyTorch 官方文档明确它不防 DoS；只读取用户显式指定的本地 checkpoint，不能自动搜索“最新”文件。

## 8. 测试与证伪矩阵

实现前先写并观察以下 RED：

1. **状态 round-trip：** tiny `nn.Module` + 两组 AdamW + LambdaLR + 真实 `CLEVRIndexStream` 完成若干合成 update 边界；保存后推进所有对象/RNG，再恢复；逐 tensor/field 比较，下一 128 个 index 和下一随机值与未中断分支相等。
2. **边界与身份：** 拒绝错误 method/seed、bool/负数、不同 dataset size、`eD+c != 128k`、非 None 梯度、scheduler 计数不符、optimizer 参数名/顺序不符、环境/schema 不符，且 live 状态不变。
3. **严格 payload：** 缺失/额外 key、非 tensor state、错误 shape/dtype、非法 Python/NumPy/torch RNG、重复/无序 CUDA devices、CUDA state 数量不符均失败；`torch.load` 必须显式 `weights_only=True`。
4. **load 事务：** 在每个应用边界注入异常，证明全部 live state 与 RNG 回滚；再注入 rollback 失败，证明不作虚假“未变化”声明。
5. **发布事务：** 正常文件可重新 `weights_only` 读取；既有 file/dir/symlink 不变；写入/link 失败无 final/staging；并发目标出现不覆盖；真实 link 后抛 `KeyboardInterrupt` 报 committed true 并保留完整 final；commit 前/后 staging unlink 失败分别报告正确 committed 状态。
6. **无别名与无哈希：** 捕获后改 live tensor 不改变 payload；发布 schema/源码没有 digest/hash 字段或调用。

最终回归先运行 checkpoint+sampler focused tests，再运行 sampler 分支已建立的 112-test CPU suite（预期 111 pass、1 CUDA skip），始终设置 `CUDA_VISIBLE_DEVICES=''`。这只是单元/数学回归，不是工程冒烟、训练或评估。

## 9. 已核实库、版本与许可

本地观测（2026-10-25，`/workspace/smbu_lt/my_envs/flightgpt/bin/python`）：Python 3.11.14、PyTorch 2.6.0+cu124、NumPy 2.2.6；`torch.load` 具有 `weights_only` 和 `map_location` 参数；当前 `torch.cuda.is_available()` 与 `is_initialized()` 都为 false。此事实只描述当前 CPU 测试环境。

采用：

- PyTorch 2.6 `state_dict`、`torch.save`、`torch.load(weights_only=True)`、CPU/CUDA RNG API，BSD-3-Clause；
- NumPy 2.2 legacy global RandomState get/set state，BSD-3-Clause；sampler 自身继续使用既有独立 PCG64；
- Python 3.11 `random`、`tempfile`、`os.link/stat/unlink`、`pathlib`，PSF-2.0。

不新增依赖，不复制第三方 checkpoint manager。现有 `run_cifar_scratch.py` 的 temp + `os.replace` 是历史 Stage 0 runner：它可作为“state_dict + weights_only”工程线索，但会覆盖同名目标、没有 sampler/scheduler/RNG 完整状态，也不满足本次 no-overwrite 与精确恢复契约，因此不直接复用。

## 10. 明确限制与后续集成门

- 本单元的 tiny-model round-trip 证明通用 state/RNG/sampler 事务，不证明真实 `CLEVRQuestionAnswerer`、固定 AdamW 参数组和学习率公式已经端到端恢复；该证明属于后续 runner 集成。
- CPU 环境不能验证 CUDA RNG；实现必须保留显式 device API 和无 GPU 路径，CUDA 回归等待用户授权的工程冒烟，不伪造通过状态。
- checkpoint 中 method/seed 是运行声明与恢复防错，不是模型确实按该方法训练的外部证明。runner 仍需把配置、日志、完成状态、逐更新样本 ID 和预测文件绑定到同一运行目录。
- 精确恢复条件于同版本软件与兼容硬件；不承诺跨 PyTorch、NumPy、Python、CUDA 版本 bitwise 一致。
- 本单元不改变模型、损失、奖励、输入或位置编码，不产生科研收益证据。

