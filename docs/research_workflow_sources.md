# 科研规则的文献与 GitHub 调研依据

核查日期：2026-10-18（Asia/Shanghai）。用途：支撑 `AGENTS.md` 和 `vln_research_workflow.md` 的设计选择。这里的流程规范是项目选择；除非明确列出评估证据，不声称它们已经提高科研效果。

## 1. 检索范围与证据边界

检索覆盖 VLN 评估、连续导航、具体架构消融，以及 GitHub 上通用科研、AI 研究和工程 agent 工作流。搜索用于发现资料，结论回到论文正文、作者项目页、官方仓库 README / SKILL.md 和 GitHub API 核对。

没有可靠的跨仓库实际调用量或用户数统计，因此无法确定“大家应用最多”的科研 skill。本次是六个相关候选仓库的关注度与适配性比较，不是全 GitHub 排名。stars/forks 不等于使用量、正确率或研究收益；最近 push 也不等于维护质量保证。

本次未选择 VLN 子任务、模型或正式实验，不宣称某个历史基线仍为 2026-10-18 的 SOTA。没有安装这些外部 skill，也没有运行它们附带的脚本。

## 2. VLN 论文：已核实内容及其适用范围

| 来源 | 核读位置 | 可支持的内容 | 本项目采用的约束 |
| --- | --- | --- | --- |
| Anderson et al., *On Evaluation of Embodied Navigation Agents*, 2018，[论文](https://arxiv.org/pdf/1807.06757) | §2、§3、Recommendations 1–2、SPL 定义 | 讨论任务定义、泛化、DONE 动作、geodesic distance 和兼顾成功/路径效率的评估；属于评估建议，不是本项目的性能实验 | 固定任务与观测条件；按停止时状态和官方距离评估；同时报告成功与效率 |
| Ilharco et al., *General Evaluation for Instruction Conditioned Navigation using Dynamic Time Warping*, 2019，[论文](https://arxiv.org/pdf/1907.05446) | §2；§3.1 / Table 1；§3.2 / Table 2 | 定义 nDTW/SDTW；在文中的人类路径判断实验中比较相关性；在 R2R/R4R val-unseen 中比较奖励设置 | 路径遵循不能仅以最终到达代替；奖励修改须落到公式与同设置实验，不能从该文推出任意新奖励有效 |
| Krantz et al., *Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments*, ECCV 2020，[论文记录](https://arxiv.org/abs/2004.02857)、[作者项目页](https://jacobkrantz.github.io/vlnce/) | 本次核实摘要及项目页任务定义；PDF 全文抓取失败，未核验数值表 | VLN-CE 使用第一视角观测和连续环境中的低层动作；任务假设不同于离散导航图 | 分开记录 graph VLN 与 VLN-CE，不直接比较分数；此条不作性能依据 |
| Chen et al., *Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation*, CVPR 2022，[论文](https://arxiv.org/pdf/2202.11742)、[会议记录](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.html) | §3.2，Eq. (3)–(4)；§4.4，Tables 2–4 | DUET 明确修改 graph-aware attention，并报告结构、训练损失和数据增强消融。Table 2 的 REVERIE val-unseen、dynamic fusion 条件下，加入 GASA 的 SPL 为 33.73，未加入为 32.88；这是论文报告，不是本地复现 | 建议须明确到算子/损失及消融；历史结果不能保证迁移收益，也不证明当前 SOTA |
| Wang et al., *What Limits Vision-and-Language Navigation?*, arXiv:2605.13328v1, 2026，[正文](https://arxiv.org/html/2605.13328v1) | §4.1、Figure 4、Table 1 | 该版本方法输入明确包含双目 RGB、指令和目标位置线索；本文仅核对输入条件，不复核其 SOTA 排名 | 比较前列出目标先验、传感器和额外信息，不能仅凭“RGB”标签视为同条件 |

规则不从这些论文推导“所有手写规则都无效”。禁止把任意阈值作为主要改进是用户明确的研究约束；官方协议边界仍须保留。可学习模块也必须经过验证，并不会因为“可学习”自动有效。

## 3. GitHub 候选快照

数值来自核查当日 `GET https://api.github.com/repos/{owner}/{repo}` 的公开响应；记录字段为 `stargazers_count`、`forks_count`、`pushed_at`、`archived`、`license`。以下仓库当时均未归档。更新时间按 API 的 UTC 日期记录；数据会变化。

| 仓库 | Stars / forks | 最近 push（UTC） | 许可证核查 | 适配判断 |
| --- | ---: | --- | --- | --- |
| [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) | 52,413 / 4,639 | 2026-10-15 | API：MIT | 采用证据批判、来源定位和按需加载思路；不整套加载与 VLN 无关的生物/医学工具 |
| [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) | 13,644 / 959 | 2026-10-11 | API：MIT | 采用 AI 研究分阶段路由和引用核验；其自动训练/循环不适用于本仓库权限 |
| [Galaxy-Dawn/claude-scholar](https://github.com/Galaxy-Dawn/claude-scholar) | 5,824 / 467 | 2026-09-23 | API：MIT | 采用原始结果审计、统计采样单位及允许/禁止表述记录 |
| [K-Dense-AI/science-superpowers](https://github.com/K-Dense-AI/science-superpowers) | 359 / 37 | 2026-09-13 | API 未识别；[LICENSE 正文](https://github.com/K-Dense-AI/science-superpowers/blob/main/LICENSE) 为 MIT | 关注度较低，但预先登记、异常诊断和质疑审查直接适配研究流程 |
| [obra/superpowers](https://github.com/obra/superpowers) | 301,321 / 26,907 | 2026-10-10 | API：MIT | 属于通用工程工作流；采用最小实现、诊断、代码审查，不把代码测试当成科学证据 |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 98,072 / 13,593 | 2026-03-26 | API：null；README 声明 MIT，未进一步确认独立许可文件 | 作为研究循环参考而非已安装 skill；采用固定评估面和记录思路，不采用自动下载、夜间训练或单一短预算指标筛选 |

以上只有前四项直接聚焦科研 skill。把通用工程或自动实验项目的关注度计作“科研 skill 使用量”会混淆类别。

## 4. 核读的具体工作流及取舍

- [Scientific Critical Thinking v1.5](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-critical-thinking/SKILL.md)：采用先提取证据、明确实验单位、保留不确定性、逐项审视混杂因素的思路。临床评估量表不直接移植到 VLN。
- [Orchestra ML Paper Writing v1.2.0](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/main/20-ml-paper-writing/ml-paper-writing/SKILL.md)：采用核实文献存在与断言相符、未验证引用显式标记的流程。文件中的通用错误率说法未经本次独立核查，不作为本项目事实引用；也不机械要求特定搜索供应商或外部 API。
- [Claude Scholar Results Analysis v0.2.0](https://github.com/Galaxy-Dawn/claude-scholar/blob/main/skills/results-analysis/SKILL.md)：采用结果/写作分离、样本独立性检查和 claim candidate 的证据边界。统计检验和图表数量按真实数据与问题决定，不照搬固定数量要求。
- [Science Superpowers README](https://github.com/K-Dense-AI/science-superpowers)：采用事前问题定义、预先登记、异常根因调查和结果质疑。用户控制实验启动和不主动监控的规则保持最高项目优先级；不执行上游自动流程。
- [Superpowers README](https://github.com/obra/superpowers)：采用工程阶段拆解及审查。它的工程测试可以证明实现符合规格，不能证明研究假设成立。
- [Autoresearch README](https://github.com/karpathy/autoresearch)：采用把评估器与可编辑方法分离的设计。本项目不采用其固定五分钟 GPU 循环作为 VLN 实验协议，亦不运行其数据准备脚本。

Kassis et al., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，arXiv:2609.00065，核读版本 v2，[论文记录](https://arxiv.org/abs/2609.00065)，摘要明确未报告任务级评估或 host selection rate。因此本次只能参考其流程组织，不能宣称该库经验证能提高 VLN 研究成功率。

这些外部内容是设计参考，不是本仓库的权限来源。设计的总体控制结构还参考了当前可用的 `harness-engineering`：固定评估面、受限可编辑面、追加证据记录、把审计与方法生成分开。本次没有创建自动运行任务。

## 5. 仍然未知

- 各科研 skill 的真实活跃用户数、调用频次及对 VLN 的因果收益：不知道。
- 用户最终要研究的 VLN benchmark、传感器设置、候选基线、可用数据/权重和预算：尚未确定。
- 截至核查日各 VLN 条件下完整的 SOTA 排名：本次未完成系统核验，不能宣称。
- Windows 连接问题的 2026-10-18 排查记录见连接手册：用户当时验证远程 `/mcp` 返回 HTTP 400，HTTP 链路可达；该记录不表示当前 Codex MCP 工具或浏览器连接状态。
