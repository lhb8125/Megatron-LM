# 四组从头预训练：设计与验收合同

## 2026-09-19 存储准入的独立组调度修复

现场A1993→2393需要连同B/C排队段预留约5.39TB，但实际约4.09TB；
D1593→1993的checkpoint更小，可能满足相同准入规则。原控制器固定选择
第一个未活跃arm，A被容量拒绝后直接等待，使D不必要地受A阻挡。
`available_allocations`现在列出通过原游标/阶段/并发校验的全部候选；
`admitted_allocation`按原ABCD顺序逐项调用完整maintenance，只跳过明确的
admit=false，选择首个获准组后提交一项，下一轮重新盘点。全部拒绝时继续监控，
不能记为complete；校验异常仍传播并停止，不能据此跳过坏checkpoint。
不修改容量预算、清理授权、保留点、训练源码/配方、5954步目标或既有作业ID。
单组串行默认及1B/10B阶段屏障保持不变。新增测试覆盖真实事件循环中A容量
受限而C/D继续提交、当前A/D候选的游标保持、全拒绝等待及完整性异常不被吞掉。
验证：24项campaign测试、全部197项预训练CPU回归及ruff通过；全量测试首次
因测试命令漏配toolkit src导入路径导致17项ModuleNotFoundError，补齐路径后
全部重跑通过，未跳过测试。生产source58b909e3...及bundle08901c99...复核未变。
宿主安全交接时确认旧控制器无子进程、watching、无intent/error，保留B7297442/
C7296632及A1993/B1593/C1593/D1593游标；v12退出后启动v13 PID1117475，
日志formal-driver-v1/driver-pressure-host-v13.log。启动后仍由原receipt/source/
bundle/历史审计及锁校验接管，不手改状态；后续实际D准入/提交另行核验。
实际验证：v13先拒绝A（free=4,093,289,422,848，required=5,386,866,535,961），
随后正常提交D1593→1993为7297954（registry 20260919-223553-1b98）；
scontrol确认64GPU/16节点/SegmentSize4/MBS2/4h/Dependency=null，PENDING。
state回到watching，无intent/error，B7297442/C7296632保持原ID。调度脚本SHA256
为2f9151ae76f2a9118fec949d4a6cbcd5ec64a3e0b80b75993b76000419f78491。
这证明实际队首容量拒绝不再阻挡另一可准入arm，不代表新D已恢复或完成50B。

最终MBS与启动验收的集中记录见[预训练启动记录](pretraining-launch.zh.md)。
本文保留按时间推进的设计、调查和实验凭据，早期状态不覆盖后续结果。

本实验独立于历史 folded-checkpoint parity。A=48 层普通模型；B=3+14×3+3
固定共享递归；C=20 层普通模型；D=3+14×3+3 动态 MoR。仅 D 注册 depth router
及使用块外 gated residual。所有组从随机参数开始，B/C/D backbone 使用相同
按参数语义名派生的 seed，不因构建顺序、EP rank 或 router 数量改变初值。

最初计划使用 OCI-HSG Qwen-tokenized SlimPajama 子集，原始存储 token 数
48,227,880,635；因来源无法核验，已按用户授权改为下面记录的约 50B 重新编码语料。
只读源文件；按完整文档 token 内容 hash 划分 0.1% validation，
相同 token 内容必定分到同一集合。训练文档固定 shuffle，串接后以 4096 切块，
GBS=2048，仅消费完整 global batches；显式 mask 排除每块末尾的 next-token target。
准备产物必须绑定完整源文件 hash、tokenizer 文件 hash 和来源核验信息。
不能把“token ID 在词表内”冒充已验证 tokenizer 来源。

默认 BF16 / AdamW，LR 3e-4→3e-5，1% token warmup + cosine，beta .9/.95，
eps 1e-8、WD .1、clip 1。MoE aux .001 对四组一致；D 另加 depth aux .001。
共享层各调用梯度累加，不除以递归次数。全部目标按 global token 归一化，
验证只统计 LM NLL，不包含 aux。

新实验入口只支持 TP=CP=PP=ETP=1；EP=16，32/64 ranks，segment=4。
历史入口保留原配置和原限制，不能为了新多节点实验移除旧 parity 的单节点 guard。
每个 EP group 的实际 rank/host/NVL domain 映射须核验，不单凭 segment 值签发证据。

D 训练使用 expert-choice。验证第 0 轮全部选中，后轮按当前 token 的
sigmoid(score)>=.5 决策，不做整段 Top-K，也不强制最少保留一个 token。
零活跃 rank 仍须参加 EP collective，不能本地提前 return 导致其他 rank 死锁。
严格因果验证必须对未来文本替换、后缀追加、逐前缀参考及空活跃集合进行测试。

MBS 1/2/4/8，每档 20 warmup + 30 measured updates。当前档整卡峰值<=80%
才继续，最终<=90% 且更新、验证、checkpoint 全部成功才合格。以吞吐选优，
与最快档差距<3% 时选较小 MBS。调优产物与正式训练分离，正式训练冻结 MBS。

正式推进 1B→10B→完整一遍；门槛按完整更新边界触发。恢复合同绑定数据、源码、
架构、并行、MBS、optimizer 和 scheduler，保存完整 RNG 与 data cursor。
证据不足时停止长跑，不自动改阈值或宣称已完成。训练结果写入单独实验报告，
不能覆盖历史技术报告或其原始产物。

## 2026-09-16 实现与运行状态

### 数据来源修订

用户在确认旧语料来源未知后，授权搜索原始 SlimPajama；找不到时从 HF 下载并
重新准备约 50B tokens。已在 OCI-HSG 可读的共享 datasets、share_data 和部分
公开 data 目录中搜索，未找到可核验的原始文本/旧预处理脚本，不能声称全盘穷尽。
HF 身份检查成功，但官方 `cerebras/SlimPajama-627B` API 返回 404。

新数据采用 `gmongaras/SlimPajama-627B_Reupload`，固定 revision
`c34c22dbb10ae6b264a2f357a909d1a537141b36`，只取 train Parquet 文件的稳定顺序前缀，
到达 50,000,000,000 个 **Qwen tokens** 后在完整文档边界停止。
这是第三方声称的原始数据重上传，尚不能独立证明与已不可访问的原仓库字节一致；
不能把它称为旧 48.23B indexed 子集，也不能隐去镜像来源。

tokenizer/config 直接来自官方 `Qwen/Qwen3-30B-A3B-Base`，固定 revision
`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`，EOS=151643，词表实际 151669，
模型输出词表为 151936（包括 padding）。不加载模型权重，不套聊天模板。
历史 Pile 实验的 vocab/merges 也已对照此官方 revision 验证，并显式用 Base EOS；
旧 SlimPajama EOS=151645 不能证明其预处理来源，故不作直接替换。

### 代码入口

| 文件/模块 | 职责 |
|---|---|
| `configs/pretraining/oci-hsg.yaml` | 四组训练合同、segment4、MBS 阈值、固定数据版本 |
| `configs/pretraining/oci-hsg-tuning.yaml` | 相同科学配置/segment4，50-update短跑使用1h预约而非正式训练24h预约 |
| `pretraining.config` | 结构、GBS/MBS、token 学习率和里程碑 |
| `pretraining.protocol` / `execution` | 显式 A/B/C/D；共享 backbone 初始化；D 因果推理 |
| `pretraining.prepare_raw` / `data` | 原始文本编码、分片恢复、hash 划分、流式索引读取 |
| `pretraining.train` | 训练/因果验证、完整 checkpoint、数据游标、全局输入 hash |
| `pretraining.tuning` / `memory` | MBS 合格判定及选择、所有 rank 的显存证据 |
| `pretraining.cost` | 独立参数计数、主矩阵 FLOPs 估算（不是硬件指令实测） |
| `pretraining.launch` | recipe → 正式 cluster-run；默认仅 render，`--submit` 才提交 |
| `pretraining.probe` / `smoke` | GB200 依赖 canary、真实 MLite 小宽度 EP16 smoke |
| `pretraining.causal_probe` | native 小模型后缀替换、全部短前缀预测、精确路由及零活跃检查 |
| `pretraining.campaign` | 只读阶段推进决策及 B−A/B−C/D−B 配对 NLL，拒绝不一致合同 |
| `scripts/pretraining/auxiliary_probe.py` | raw auxiliary gradients 的独立 FP64 DP/EP 归约参考、microbatch 累积及 optimizer 精确分片覆盖检查 |
| `scripts/pretraining/indexed_resume_probe.py` | tiny backbone、官方词表、真实 indexed 输入、seq4096/GBS2048 两步外部进程恢复检查 |
| `scripts/pretraining/inspect_smokes.py` | 只读核对各 rank 初始化、B/C/D backbone、参数量及 smoke 产物；不签发正式证书 |
| `scripts/pretraining/shared_gradient_probe.py` | B共享参数梯度 vs 48个独立层对应梯度之和；同native内核，仅验证weight-tying链式法则 |
| `scripts/pretraining/probe_suite.py` | 单allocation顺序启动各arm及外部恢复进程；任一失败立即停止 |
| `scripts/pretraining/inspect_trials.py` | 核验50次更新/全rank显存/验证/checkpoint/全局输入hash后判定下一MBS；未扫完不冻结 |
| `scripts/pretraining/initialization_probe.py` | 官方完整宽度四组的参数计数和逐参数初始指纹；不执行训练，不冒充完整显存峰值 |
| `scripts/pretraining/issue_acceptance.py` | 汇总真实逐rank证据、完整数据/环境/模型指纹、完整宽度选定MBS恢复及MBS扫描后，才生成四组正式验收凭据 |
| `scripts/pretraining/drive_tuning.py` | 顺序推进四组pilot的MBS扫描；接管已有作业、审计完成产物后才提交下一档；不签发证书或启动正式训练 |

MoE auxiliary 沿用 native 的每 rank / microbatch 负载均衡统计与 AutoScaler；
梯度按 microbatch/DP 平均。该统计不是先收齐整个 GBS 再统一计算的全局路由统计，
改变 MBS 会改变统计粒度；不得声称已证明其对 MBS 完全不变。
D depth auxiliary 为原先逐轮候选 token 的均值。正式验收仍需梯度与分布式归一化实证。

验证覆盖末尾不满 4096 的序列，不丢弃验证尾部。各 chunk 最后一个位置没有本
chunk 内的 next-token target，显式 mask；报告输入 tokens、有效 targets 和边界数量。
空余评估 rank 运行不计分的 dummy，以保持 EP collective 的调用顺序。

`train --mode tune` 是独立的 50-update 性能短跑：不允许 resume 或改变停止点，
不算正式训练 tokens，前 20 次 warmup、后 30 次计吞吐，包含验证和 checkpoint。
整卡显存按 20 ms 间隔采样 CUDA mem_get_info，并独立记录 allocator 峰值；
采样可能漏掉极短暂的外部分配峰值，该限制须随结果报告。
MBS审计入口读取完整产物，核对最后30步吞吐、每个rank峰值、初末验证和完整
checkpoint游标；不同MBS必须保持同一合同（仅micro_batch_size可变）及50个全局
输入hash。低显存MBS1单档只返回下一档2，不能被标为已冻结。完整但>90%显存
的档位保留记录而不入选；没有完整产物的失败/OOM需要先诊断，不伪造成功记录。
独立pilot driver在CPU端每30秒读取SLURM实际状态，整个流程最多一个GPU作业；
不因排队自动重复提交。每档成功后下载JSON证据（不复制巨型checkpoint权重），
审计全部50次更新、显存、验证和checkpoint完成标记，再按1→2→4→8及A→B→C→D
推进。跨档/跨arm数据、环境、scheduler、全局输入hash必须相同；源代码或配置改变
立即停止后续提交。driver使用排他文件锁、原子状态文件及提交前持久化intent；
提交结果不确定、SLURM失败、OOM或验收失败均停止，不能在重启时自动重复sbatch。
这是一项有界实验执行进程，不是自动AI诊断服务；失败仍需调查修复。该流程的pilot
选择不能代替最终50B数据的MBS验收，不会自行推进正式训练。
`--mode train` 必须提供绑定源码、数据、环境和完整配置的真实验收证据，
不得手工把未完成项改成 true。目前没有签发正式四组验收证书。
新增凭据汇总入口尚未执行：要求四组tiny语义smoke、auxiliary、共享梯度展开、
完整宽度初始化，以及最终数据/选定MBS的完整宽度外部恢复证据。MBS扫描须完成，
所用50B语料、tokenizer文件hash、数据来源revision和跨MBS/跨arm输入hash都需匹配；
pilot或未完成的扫描不能用于签发。生成前在CPU上重验全量indexed文件hash。
凭据保留各项证据文件hash及scope，不把tiny语义测试描述为full-width数值证明。
正式/tuning 入口另外检查官方 30B 的层数、hidden/head、expert、词表和 EOS，
拒绝把小宽度 smoke 配置当作正式模型。跨进程 smoke 用 `--resume` 指向旧 smoke
目录，比较 model、真实 FP32 master、Adam moments/step、RNG 的逐 rank 指纹，
并比较下一次更新与不中断运行；它使用两步合成数据，不能冒充真实 indexed-data
跨进程验收。相应 GPU 结果未产生时，这些测试代码不计作通过。

### 已取得的证据（不是正式训练结果）

- `7192235`：OCI-HSG 基础环境探针，COMPLETED；GB200，PyTorch
  `2.11.0a0+a6c236b9fd.nv26.03.46836102`，CUDA13.2，TE `2.14.0+f031cf8`。
- `7192400`：固定 native commit 的真实 MLite 导入及 TE BF16 forward/backward
  canary，COMPLETED；native Python tree SHA256
  `6e235d0f9097fb8b12e794f6ef0ac5e86c72747df2275c3043c9471a3449d5d3`，772 个文件。
  容器与历史 H100/torch2.10 环境不同；此 canary 不等于完整训练验收。
- `7192345`：CPU 数据准备在下载前因 `.env` 根目录识别错误失败；已修复
  cluster-run 的显式项目根传播，不通过伪造 vendor 目录解决。
- `7192425`：修复后 10M 数据准备 COMPLETED，耗时 2m35s；10,003,537 tokens，
  9,816 documents，train=9,999,162，validation=4,375，EOS151643。
  bin SHA256=`0c82b8ae2def6f3c4cad2aab07cc9fceee3f212b21e929c85939730da3ab9a14`；
  idx SHA256=`2f084789d1c3d35b2ca01ed0aea74f9b203bca5891c8534947657c9ed291607e`。
- `7192513`：约50B数据准备，32 CPU、无GPU、24h limit；后因HF读取超时FAILED，
  已封存14个分片。修复和恢复作业见下方2026-09-17记录。
- `7192545`：B 小宽度、32 GPU / EP16 / segment4 的真实多节点 smoke 在模型
  构建前失败：固定 native 的 checkpoint stack 要求 nvidia-resiliency-ext>=0.6.0，
  容器原版为0.5.0；未执行训练或通过拓扑检查。
- `7192712`：独立 experiment venv 补齐 nvidia-resiliency-ext0.6.0 后，完整训练
  protocol 导入和 TE canary 完成，但 pip 报 protobuf/wandb 冲突，未作为正式环境。
- `7192850` / `7192997`：严格 `pip check` 发现容器自带 quack-kernels0.3.9
  要求 cutlass-dsl>=4.4.2，以及 CUDA13 companion library 必须与 base 同版。
  bootstrap 显式固定 `nvidia-cutlass-dsl[cu13]==4.4.2`、grpcio/tools1.78.0、
  protobuf6.33.6；共享镜像未修改，检查未跳过。`7193036` 已通过 `pip check`、
  完整训练 protocol 导入及 TE canary；这是候选环境证据，仍非完整训练验收。
  环境记录：`runtime/mor-pretraining/gb200-environment-pinned-deps-v3.json`。
- `7193093`：修订独立环境下重跑 B，32 GPU / EP16 / segment4，输出
  `runtime/mor-pretraining/smoke-b-ep16-v2`；模型构建前失败：把适配层名 `te`
  误传为 native attention 算法选择项。修复为显式 `fused`，静态 API 回归直接读取
  pinned native 的 selector 表，拒绝复发；不修改 native 的合法值检查。
- `7193288`：下一次 B 重跑暴露启动器遗漏包内 `configs/`，初始化引用 parity
  参数语义名 helper 时缺少 `topologies.json`。启动器改为同步包含 `src/mor_mlite`
  和完整 `configs/` 的内容寻址快照，不再执行远端可变工作树；快照 hash 同时绑定
  源码和配置。CPU 测试覆盖缺失配置、快照复用和已冻结文件被修改时拒绝。
- 新入口现显式调用 native `set_deterministic`，在 CUDA 工作前配置 PyTorch/TE
  确定性开关，关闭 TF32；仅 `ImplConfig.deterministic=True` 不足以启用这些开关。
- `7193306`：用封存的第一个完整 indexed 分片准备独立 performance pilot；
  只读原分片、重验 hash/EOS、使用同一 tokenizer 和划分算法。用于 correctness/MBS
  短跑，不等待全量50B编码完成，不计正式 tokens。manifest 标记
  `performance-pilot-only`，正式 `train` 入口拒绝此用途。全量数据任务继续独立运行。
  该作业已 COMPLETED（2m32s），2,378,284,596 tokens / 2,361,579 documents；
  train=2,375,813,531，validation=2,471,065。bin SHA256
  `a8002f134f1cbcacee71edd5892527591f906fa67e3050b6c5a15f894ceb4fe4`；idx SHA256
  `2a3aa166cdddfca73f1afcfe314671d75729ab5f101c59d1d16cdb282e2fd144`。
  路径 `runtime/mor-pretraining/data-pilot-shard0`，正式30B配置/EOS检查也已通过。
- `7193374`：B 的修订小模型 smoke，32 GPU / EP16；固定 package bundle
  `a41e4b651db979637305f2bcf909c7b26e8693ea560e49f2d7e1113fe09ff482`，
  输出 `runtime/mor-pretraining/smoke-b-ep16-v4`，COMPLETED（2m18s）。全部32rank
  前反向/更新、MoE auxiliary 非零梯度、因果短前缀检查和 checkpoint 保存通过；
  prefix prediction 最大绝对误差0。实际 EP domains=`block110`、`block127`，
  每组16ranks/4nodes位于同一domain。dense 副本和 expert-DP 副本初始化bitwise一致；
  小模型64,297,344独立参数，32rank FP32 master shards合计同样64,297,344元素，
  没有因递归调用重复注册optimizer参数。仍非30B正式验收。
- `7193505`：B 另一次32rank进程启动恢复，COMPLETED（2m07s）；model、FP32
  master、Adam moments/step、RNG指纹，以及下一次更新/输入/游标与不中断参考
  bitwise一致。这是tiny两步合成数据的连续性证据，真实indexed数据恢复仍需验收。
  package Python source SHA256=`d54434fb1541bfd8f1ca7a88b80c318cc9cdf31fca73195a112773ab77667b6e`。
- `7193565`：C 的32rank smoke，COMPLETED（2m07s），同一固定源码和候选环境。
  全部rank更新/因果检查通过，最大短前缀预测误差0；B/C的完整backbone初始化
  bitwise一致，独立参数量均64,297,344，optimizer master全局元素数相等。
- `7193665`：D 的32rank smoke全部rank产物已通过：前反向/optimizer更新、MoE和
  depth auxiliary非零梯度、未来替换及逐前缀预测/路由检查。最大短前缀误差0，
  全局累计178次零活跃rank/round，无collective死锁。D与B/C的backbone初始化
  bitwise一致；D独立参数64,297,728，比B/C仅多384个depth-router参数（tiny hidden128）。
- `7193769`：A 的32rank普通48层 smoke，COMPLETED（2m23s）；全部rank产物通过。
  至此四组 tiny-width / EP16 训练更新、短前缀检查、checkpoint保存均有实测证据；
  这不替代完整30B模型、长序列和真实数据的验收。
- `7194935`：官方完整宽度的四组初始化，64 GB200 / EP16，COMPLETED（2m59s）。
  全部256份rank指纹已核验：dense/expert-DP副本bitwise一致，不同expert初值不同，
  B/C/D的完整backbone bitwise一致。独立参数实测：A=30,532,122,624；
  B/C=13,084,744,704；D=13,084,750,848（额外6,144个depth-router参数）。
  官方config SHA256=`7e4142150e976c6b4796adf88fce0a2a23e581ed63bcede7e1d69a645e73b362`。
  这里只分配/初始化模型，不是forward/backward及Adam状态后的显存峰值，不能用来选MBS。
- `7195074`：A完整宽度/64GPU/EP16/MBS1/seq4096/GBS2048的pilot性能短跑已提交，
  固定20warmup+30measured、含初末验证和checkpoint。新增短跑recipe仅缩短预约至1h，
  科学配置和TUNING字段与正式recipe完全相同。package bundle
  `e6d288229408b2d78bdcf8c3b8507f0f1b4c24d42af8553c067ebdfdabbcf2f3`，
  与a41快照相比只多一个tuning recipe；Python源码SHA仍为d54434...。
  pilot结果只用于早期显存/吞吐验证，不算正式tokens，不签发最终数据验收凭据。
  CPU顺序执行器已接管此作业；状态/日志位于主项目
  `runtime/mor-pretraining/pilot-driver-v1/{state.json,driver.log}`。它会逐档核验后
  原作业失败后driver按合同停止，没有继续提交B/C/D；修正版作业和新driver见下方记录。
- 独立 auxiliary probe 以 finalize 前 raw gradients 的 FP64 replica sum / dense-DP
  world 为参考，分别检查 dense 和 expert 参数；并对所有 optimizer shard 的参数
  区间检查全局无遗漏/无重叠。1 vs 2 identical microbatches 仅验证累积缩放，
  不证明不同 MBS 下真实不同样本的负载均衡统计相同。GPU结果尚待执行。
  四组专项作业 `7193819` 中A/B/C的全部32rank已通过；相对FP64参考的最大
  relative-L2分别为6.44e-9、5.86e-9、5.42e-9。D尚未进入模型计算就因
  TCPStore `EADDRINUSE` 失败：启动器选用了38822；集群login节点的系统临时端口
  区间为9000–65000，连续子进程的通信listener存在竞争风险。启动器现读取实际
  节点的临时端口下界，在其下方选择互不重复的非特权端口；不足安全范围则报错。
  此失败不算D数值失败，D单独重跑；A/B/C产物不丢弃也不冒充整作业成功。
  修复后D作业 `7194119` COMPLETED（2m04s），全部32rank通过。同一个模型源码
  `d54434...`、auxiliary probe SHA256
  `13b57a8343e121a3a6335add67c83c263c250e9bd7cc93e421c52c49e1102b07`。
  四组auxiliary归约/重复microbatch缩放/optimizer精确分片覆盖均取得GPU证据。
- 新 indexed resume probe 直接使用生产 TokenStream、packing、GBS、序列长度和
  token scheduler；仅缩小 backbone 宽度，词表151936保持真实Qwen输入可读。
  保存前显式推进 Python/numpy/CPU/CUDA RNG，并要求新进程初始RNG不同，防止
  无效 RNG 恢复被初始相同seed掩盖。比较恢复后的model/master/Adam/RNG、学习率、
  下一完整global batch的hash、游标及更新。
  四组按A→B→C→D及每组全新恢复进程顺序运行，作业 `7194324`，seq4096/GBS2048。
  初始2h预约排队预计延迟较长；pending期间将此tiny两步验收的SLURM时限缩短为
  30min（scheduler override；原生成脚本仍记录2h）。不改变科学配置；若超时，
  不完整arm不计通过，保留已完成的逐rank证据后再定位耗时。
  本地另补显式 `--full-width --mbs <selected>`，用于完整宽度、选定MBS的同类两步
  验收（先严格核验官方base config），而非绕过正式train验收门槛；tiny仍默认MBS1。
  该本地扩展不热更新已排队 `7194324` 的脚本；完整宽度分支尚未执行。
  `7194324` COMPLETED（18m26s）：A/B/C/D已经完成原始两步及独立进程恢复，
  共128份rank凭据的model/master/Adam/RNG/下一更新bitwise通过；游标均16,777,216，
  四组第二global batch SHA256一致：
  `5526158b4f919c574d1ceb753a970666659ce00a7175d6f72e46e53acb09abad`。
  运行中的原probe SHA256=`45ad1094d609cfbe746edb548474ea1099fd5cce6896f96ccfb5b9544fe0d9af`。
  原probe和suite脚本已归档到主项目 `runtime/mor-pretraining/evidence/probe-sources-7194324/`，
  并复核脚本hash与实际凭据一致，避免后续脚本扩展破坏历史可复现性。
- 共享梯度probe用实际B参数复制到48层普通stack的独立Parameter，显式映射
  三次共享调用（不调用待测layer_indices），比较LM loss及所有对应梯度之和。
  每参数tensor similarity/cosine阈值均0.999，用于BF16梯度累加；不验证native
  Transformer内核数学，也不把同内核展开称为独立PyTorch模型参考。
  `7194187` COMPLETED（2m07s），全部32rank通过，14,816个逐参数梯度检查；
  最小tensor similarity=0.9999907854，最小cosine=0.9999919668，LM loss差值0。
  probe SHA256=`386f1a18f271f3947c783ecebb580f0c24336d1909b54cc79da3d103e25ecd00`。
- 新增 CPU 合同/数据/因果语义测试 30 passed（含因果 probe 对人为未来泄漏的拒绝测试）；
  连同历史 objective/CLI/receipt/checkpoint 回归为
  加入27个probe guards/子进程顺序/共享映射/MBS审计/凭据guard/顺序执行器测试后为
  152 passed、2 skipped；后两项需要native CUDA环境，未冒充CPU通过。
  这使用独立的
  torch2.5.1 CPU 语义测试环境，不代表 GB200 native 测试。
- cluster-run 回归 19 passed，覆盖 segment 经模板解析和远端传递后仍输出，
  以及没有 Megatron vendor 时的项目根传播。

尚未完成：四组 full-width native 完整验收、
完整宽度/选定MBS的indexed数据跨进程恢复、各组 MBS 扫描与冻结、正式 1B/10B/全程
训练及配对结果报告。当前不能宣称复现了效果提升或完成了本计划。

### 2026-09-17 两项失败修复（64卡50-step验证通过）

- 数据作业 `7192513` 在7h43m后因HF `ReadTimeout(read timeout=10)` 失败，
  不是tokenizer/hash检查失败。14个完整indexed分片仍在，未删除或覆盖。
  `prepare_raw` 对连接/读取超时、HTTP408/429/5xx做最多8次有界指数退避，
  download/etag读取时限分别60/30秒；401/403/404和数据合同/hash错误立即失败。
  日志不打印异常URL，避免暴露签名。包依赖显式声明requests，不依赖
  huggingface-hub的间接安装来提供异常类型。恢复先核对已完成分片合同和bin/idx
  hash，再复用，不再为这些分片重复调用HF下载。新作业 `7205746` 已运行，
  仍使用原50B目标、固定revision/tokenizer/文档顺序，不改变数据科学合同。
  恢复实测14个sealed shards逐个hash通过并复用：33,448,142,599 tokens，
  33,062,106 documents。不是从头重新编码，也没有跳过完整性检查。
  恢复任务之后已完整编码并封存新分片0014，继续处理0015；约50B全量准备尚未完成。
- GPU作业 `7195074` 实际完成初始验证及3次更新；第4次更新期间失败。
  首三个LM loss为12.38604726、11.08348238、12.20278732。这只是pilot，
  不计正式训练tokens。日志显示rank0–15等已在default NCCL组做3元素loss统计归约
  （seq26），而EP组rank32–47仍处于MoE all-to-all/下一层all-gather
  （seq36572/36573）；600秒后watchdog超时。没有观察到OOM或Python模型异常。
  日志定位到不同NCCL组在rank进度不均时的交错顺序问题；下面的修正版在同配置
  完成全部50次更新，支持这一定位。此结论限于本次故障，不声称排除了所有底层
  NCCL故障或证明所有硬件/并行配置无死锁。
- 生产训练新增独立 `HostControl`：loss/耗时/验证统计先detach并复制到CPU，
  在Gloo组归约；输入hash、显存收集、控制广播/barrier也显式用该组。
  模型EP/DP通信、梯度、optimizer、精度和LR配方不变，不延长NCCL超时来掩盖挂起。
  历史smoke/evaluate调用保留兼容默认值，正式/tune入口显式使用HostControl。
  修正版作业 `7205862`（registry `20260916-181330-33d3`），同64GPU/EP16/
  MBS1/seq4096/GBS2048，输出 `runtime/mor-pretraining/pilot-fix1-a64-mbs1-v1`。
  新driver状态独立保存在 `runtime/mor-pretraining/pilot-driver-fix1/`，旧失败记录保留。
  修正版已实际通过原失败的第4步（36.48秒，LM loss=10.57215746），并完成第5步；
  前三步全局输入hash与旧作业逐步一致。最终作业COMPLETED，退出码0，耗时33m33s；
  50次更新、step50 checkpoint及初末严格因果验证全部成功。只读审计另核对全部
  50个更新记录/输入hash、64rank显存记录、完成标记及checkpoint游标419,430,400。
  末步训练LM loss=7.59120215；固定pilot验证NLL由12.38995205变为7.61852597，
  PPL=2035.55944，有效targets=2,470,461。此结果不是四组效果比较。
  后30步吞吐235,251.92 tokens/s；64rank最大整卡峰值94,864,080,896 bytes
  （47.9359%），最大allocated/reserved峰值分别80,265,787,392/83,709,919,232 bytes；
  每rank至少92,041次显存采样。整卡20ms采样仍可能漏掉极短峰值。
  SLURM实际分配GPU-hours=35.7867；应用计时GPU-hours=33.9250，两者口径不同。
  证据在主项目 `runtime/mor-pretraining/pilot-driver-fix1/evidence/7205862/`。
  MBS1通过且峰值低于80%，driver审计后进入A/MBS2提交流程；尚未冻结MBS，
  仍不计正式tokens，也不签发最终50B数据验收凭据。
- 新增下载及HostControl测试（含真实双进程Gloo归约/对象收集/广播/barrier）后，
  完整CPU回归为164 passed、2 skipped。
  修订Python源码SHA256=`58b909e36cc88bea5ce1efc1e8f400c4f5597bd5384c1ba631019c72df62359a`，
  含配置快照=`08901c997f8058ab6e33f52d1d585aaee4fc668ee6ba847ed3e527b0747d0b69`。
  旧快照及失败driver状态保留，不伪造完成记录；源码SHA已变化，旧验收凭据
  不会自动转为新版本通过证据，最终正式启动仍需完整绑定新版本的验收。

### 2026-09-17 全量数据完成与正式数据MBS扫描

- CPU恢复作业 `7205746` 已COMPLETED（4h27m33s），最终50,000,007,401
  stored tokens / 49,434,783 documents；train=49,949,241,482、
  validation=50,765,919。正式数据路径为 `runtime/mor-pretraining/data-50b/prepared`。
  bin SHA256=`a09a35ad6da4b4ee212d45bbb94823b8e9de4bde200396ae0aac62c0f6436b70`，
  idx SHA256=`d701eca069b82171449bf1c6947bc36d4f541cd4dc78e4c3af0a2736e03de677`。
  上方“全量尚未完成”是此前时间点记录，现以本段完成产物为准。
- pilot A/MBS2 `7206503` 已完成20+30步，后30步369,115.64 tokens/s，
  最大整卡峰值77.37696%，初末验证和checkpoint通过。相比MBS1吞吐提高56.90%。
  pilot MBS4在提交前rsync阶段失败：远端Lustre对native
  `megatron/core/tokenizers/text/__init__.py` 的stat返回EIO；未提交GPU作业，
  不能记为OOM或模型失败。2026-09-17再次stat/读取正常，与本地单文件SHA一致；
  整个native树772个Python文件SHA仍为 `6e235d0f...d5d3`。未删除文件、
  跳过检查或替换native版本，原失败状态保留。
- 数据就绪后不再继续pilot扫描。新作业 `7213156`（registry
  `20260917-020437-0900`）从全量数据A/MBS1开始，64GPU/EP16/segment4，
  seq4096/GBS2048，仍20warmup+30measured，初末验证使用完整固定验证集。
  输出 `runtime/mor-pretraining/finaldata-a64-mbs1-v1`；顺序执行器位于
  `runtime/mor-pretraining/finaldata-driver-v1`。旧pilot数据、schedule、输入顺序
  不同，故不复用其成功档位作为最终验收；新扫描同样不消费正式训练tokens。
- 完整宽度indexed恢复探针显式使用生产 `HostControl` 做输入hash收集及保存
  控制同步，避免继续用default NCCL组交错统计。只改独立probe，不改变生产
  package source/bundle，旧probe产物保留；相关CPU回归31 passed，ruff通过。
  此测试修改尚不等于GPU恢复验收通过。正式启动仍等待新版本全套凭据。

### 2026-09-17 复用pilot筛选、最终数据复验候选

用户指出MBS2已经通过，全部从1重跑缺乏必要性。此前把“最终候选必须在最终
数据上验收”扩大为“所有已成功筛选的较小档位必须换数据重跑”，条件过严。
`7213156` 在启动1m27s、尚无训练完成记录时主动取消，未作为失败或通过档位。
保留原作业和driver-v1记录，没有伪造50步完成。

新 `tuning_evidence.py` 明确区分screening与qualification：

- `--screening-run` 仍须通过原完整50步审计，绑定相同源码、模型、并行、GBS、
  序列长度、seed、优化器和环境；各screening档位之间的数据/日程/输入hash一致。
- 最终数据从screening选出的候选开始复验，再按该次实测显存决定是否继续增大；
  同一MBS的新实测覆盖旧吞吐/峰值。不会因为pilot峰值低而忽略新数据的80%停止线。
- 最终冻结的候选必须在最终数据完成50步、完整验证和checkpoint且峰值<=90%。
  如果比较后pilot-only档位成为最快候选，必须补其最终数据复验，不能直接签发。
- 所有最终数据档位/arm的全局输入hash严格一致；pilot输入hash单独保留，不冒充
  最终数据一致性。跨数据吞吐只用于筛选候选，不能称为同输入严格性能对照。

A/MBS2最终数据复验作业 `7213308`（registry `20260917-021220-85d5`），
输出 `runtime/mor-pretraining/finaldata-a64-mbs2-v1`，仍64GPU/EP16/segment4。
新driver `runtime/mor-pretraining/finaldata-driver-v2` 读取并审计pilot
`7205862`/`7206503`证据，接管此作业；后续按A→B→C→D顺序执行。
包内生产源码/配置快照不变，只修改独立审计及执行脚本，旧验收不自动转为通过。
另扩展probe suite的smoke模式，在单allocation内逐arm启动独立进程及端口，
减少重复调度开销；因果/auxiliary/恢复检查本身不删减。CPU故障注入覆盖筛选合同
拒绝、最终数据显存停止、未复验候选拒绝冻结、重复档位及smoke真实topology参数。
本轮完整CPU回归170 passed、2 skipped（native/CUDA专用项），ruff通过。
生产Python源码SHA仍为 `58b909e3...359a`，与已通过pilot及新GPU任务一致。

### 正式campaign顺序执行器（尚未执行）

新增独立 `scripts/pretraining/drive_campaign.py`，默认仅核对四组已签发凭据并
打印下一阶段，必须显式 `--execute` 才提交。缺任何一组证据、issuer/source/
data/config/environment不匹配均拒绝；不替代 `issue_acceptance.py` 的真实GPU验收。
每次提交前另核对远端验收文件SHA与本地已检查文件一致。

执行顺序为全部A/B/C/D到1B，再全部到10B，最后完成一遍。实际边界为120和
1193 updates；最终5954 updates。默认每allocation最多1000 updates以便长阶段
可在24h预约内分段，分段仅改变停止位置，保留完整5954步token LR日程及数据游标，
通过上个产物的实际checkpoint恢复optimizer/RNG。每次只保留一个活跃训练作业，
不提前同时提交四组。每段实际耗时是否适合预约仍需以正式吞吐核查，1000不是
所有模型都必定可在24h完成的保证。

成功后审计SLURM状态、所有更新序号/token/LR/有限loss和梯度、验证里程碑、
checkpoint完整标记/游标及全rank显存记录；跨arm重合更新的全局输入hash须一致。
状态原子写入且加排他锁，提交前记录intent；失败或提交不明确立即停下，不能
自动重复sbatch。新增CPU故障注入9项通过，覆盖阶段顺序、分段、缺失验收、
重复更新、重置LR、非有限值、缺checkpoint/验证及含糊提交结果；这不是GPU训练结果。

另新增独立 `inspect_rejected_trial.py`，为后续可能的容量边界失败保留明确的数据
类型：`RejectedTrial` 永远不合格，不含未测得的吞吐/峰值。审计要求原始作业
manifest、terminal FAILED/OUT_OF_MEMORY且非零退出的scheduler记录、实际
`torch.OutOfMemoryError: CUDA out of memory.` 异常日志及各文件SHA；host OOM、
NCCL timeout、主动取消、警告或仍在运行的作业不能据此认定CUDA容量失败。
此入口尚未生成真实拒绝记录，也未改变运行中driver遇失败停止的行为。成功档位
仍须原50步完整审计，不能将未完成的失败档位转换成成功结果。
本轮完整CPU回归182 passed、2 skipped，相关独立脚本ruff通过。

### 2026-09-17 A/MBS2最终数据50步复验通过

`7213308` 已完成并通过driver及独立只读审计：50个更新/全局输入hash、64rank
显存、完整step50 checkpoint和初末全量验证。最终语料manifest SHA256为
`fbbeba8051302c86818f4edd8d95131fb61a579bbbfa5f0ed5d54def7e7d6153`。
全量训练horizon=5954 updates / 49,945,772,032 input tokens，统一舍弃尾部
3,469,450 tokens；本次仅性能短跑419,430,400 tokens，正式训练tokens仍为0。

后30步吞吐366,358.634 tokens/s；整卡最大144,574,971,904 bytes（73.05544%），
allocated/reserved最大125,659,956,224 / 133,164,957,696 bytes；64rank每个至少
76,399次20ms显存采样。应用计时1573.6868秒、27.97665 GPU-hours（不含容器启动）。
末步训练LM loss=7.61145356；完整验证集NLL由12.38616170变为7.58769781，
PPL=1973.76429，有效targets=50,753,524，输入tokens=50,765,919，排除12,395个
chunk末尾预测目标。不是四组效果比较，也不能与pilot不同验证集的NLL作配对比较。
峰值低于80%，扫描继续A/MBS4，MBS仍未冻结。
本地证据：主项目 `runtime/mor-pretraining/finaldata-driver-v2/evidence/7213308/`。

### 2026-09-17 A/MBS4容量边界与A组选择

A/MBS4 `7213756`（registry `20260917-024539-e6e0`）FAILED，退出143，5m30s。
先完成全部验证集初始评估，NLL=12.386161704331121，与A/MBS2初始值一致；
第一次训练更新的forward尚未完成就CUDA OOM，不是50步性能结果。
首个异常位于固定native `linear_cross_entropy.py` 的
`_vocab_parallel_entropy -> torch.exp(shifted)`。MBS4/4096/151936词表的FP32
完整词表临时张量约9.27GiB：rank30剩余4.87GiB、未使用reserved3.55GiB；
rank25剩余264.44MiB、未使用reserved2.63GiB、进程使用184.04GiB/容量184.31GiB。
至少这些rank即使回收所有未使用cache也不足以满足本次分配，不能单纯归为碎片。
后续NCCL/step终止是此次CUDA异常之后的日志，未把143本身当作OOM证明。

本次不改native entropy实现、BF16配方或启用额外recompute以强行让某档通过；
结论限于当前固定实现的MBS容量，不声称模型在其他优化实现中不可能使用MBS4。
`record_rejected_trial.py` 从真实registry和terminal sacct获取原始提交、manifest、
错误日志与文件hash，封存于 `finaldata-driver-v2/evidence/7213756/`。
`RejectedTrial` 无吞吐/峰值伪值，永远不合格；与成功结果冲突、来源改变或非CUDA
失败均拒绝。原driver失败状态归档为 `failed-7213756.json`。

driver仅在显式 `--reconcile-rejected 7213756` 后，重新核对SLURM实际终态、
拒绝证据、原扫描顺序和科学合同，才推进到B/MBS1。未绕过任一成功候选的50步
完整验收；issuer对拒绝档位同样不要求不存在的50个输入hash，但最终选定MBS必须
拥有最终数据上的成功证据。A组已完成当前扫描，选择MBS2（366,358.63 tokens/s、
73.05544%整卡峰值），不继续MBS8。四组正式凭据仍未签发。
本轮完整CPU回归183 passed、2 skipped，ruff通过。

### 2026-09-17 B组启动前传输恢复

A组扫描结束后，B/MBS1的第一次提交在 `sync_project_to_remote` 的rsync scripts
阶段被SSH连接关闭（rc255），发生在生成registry ID和远端sbatch调用之前，没有
新的GPU作业。Toolkit `cluster_run.remote.sync_with_retry` 现在仅对已明确的
提交前SSH断连及rsync EIO进行最多3次有界重试；不重试认证/缺文件/源码校验错误，
也绝不自动重试可能已成功的sbatch。26项cluster-run回归通过。

修复后B/MBS1重提中实际又出现一次同步断连，第二次同步成功并提交
`7214001`（registry `20260917-030250-4f0b`），仍相同64GPU/EP16/segment4/
BF16、固定生产源码58b909e3...、最终50B数据和50-step tuning。
输出 `runtime/mor-pretraining/finaldata-b64-mbs1-v1`。
driver通过显式 `--adopt-submitted` 核对新registry命令与原intent、重审已完成A组
证据后接管；归档原失败状态，不手改状态跳过检查，不重复提交已存在的作业。

### 2026-09-17 B/MBS1最终数据短跑通过

`7214001` COMPLETED、退出0:0，SLURM耗时37m58s。driver与独立
`inspect_trials.py`均核验50次更新、初末完整验证、step50 checkpoint以及64rank
显存产物。后30步吞吐233,416.819 tokens/s；整卡峰值81,222,107,136 bytes
（41.04249%），allocated/reserved峰值67,746,406,912 / 70,019,710,976 bytes；
每rank至少105,408次20ms采样。应用计时2173.2427秒、38.63543 GPU-hours。
验证NLL由12.38149528到7.66464100，PPL=2131.62740；有效targets=50,753,524。
第50步训练LM loss=7.66399583。完整50步全局输入hash和LR与最终数据A/MBS2逐步
一致；本次消费419,430,400个调优input tokens，正式tokens仍为0。

生产源码58b909e3...、最终数据fbbeba80...、环境487e6368...未变，配置仍
64GPU/EP16/segment4、GBS2048/seq4096。峰值低于80%，下一档为B/MBS2，
本档成功不能冻结B的MBS或用于宣称四组效果结论。证据保存在主项目
`runtime/mor-pretraining/finaldata-driver-v2/evidence/7214001/`。
driver随后实际提交B/MBS2 `7214574`（registry `20260917-034527-3f04`），
输出 `runtime/mor-pretraining/finaldata-b64-mbs2-v1`，仍为独立50步调优任务；
没有因一次状态读取SSH rc255而重复提交或把成功B/MBS1重跑。

### 2026-09-17 B/MBS2最终数据短跑通过

`7214574` COMPLETED、退出0:0，SLURM耗时27m49s。driver和独立审计均通过
50次更新、全量初末验证、step50 checkpoint、64rank显存及B/MBS1与MBS2的
50步输入hash/科学合同一致性。生产源码、数据、环境与B/MBS1相同。
后30步吞吐364,174.451 tokens/s，比B/MBS1高56.02%；整卡峰值139,990,597,632
bytes（70.73890%），allocated/reserved峰值122,467,960,832 / 128,836,435,968
bytes；每rank至少75,405次20ms采样。应用耗时1560.3479秒、27.73952 GPU-hours。
初始验证NLL=12.381495275521884，与B/MBS1相同；最终验证NLL=7.57927797、
PPL=1957.21528，第50步训练LM loss=7.59060973。各档独立调优，不计正式tokens；
不能将这50步的NLL差异当作模型效果结论或跨MBS完全等价证明。

最大整卡显存低于80%，因此下一档是B/MBS4，B尚未冻结；MBS2仅为当前最优合格
候选。完整证据在主项目 `runtime/mor-pretraining/finaldata-driver-v2/evidence/7214574/`。
B/MBS4随后实际提交为 `7215099`（registry `20260917-042121-7d01`），输出
`runtime/mor-pretraining/finaldata-b64-mbs4-v1`。提交前一次瞬时rsync失败经有界重试
恢复，未重复提交GPU作业；本条仅记录启动，不代表MBS4通过。

### 2026-09-17 B/MBS4容量边界与B组选择

`7215099` FAILED、退出143:0，耗时6m16s。初始完整验证NLL仍为
12.381495275521884，与B/MBS1和MBS2相同；首个训练forward发生CUDA OOM，
没有完成任何训练更新。堆栈定位到固定native `linear_cross_entropy.py` 的
`_vocab_parallel_entropy`：多rank在第33行 `(exp_logits * logits).sum(dim=-1)`
分配9.27GiB完整词表FP32临时张量失败，另有rank在第29行 `torch.exp(shifted)`
失败。rank47可用6.06GiB、未使用reserved1.50GiB；rank6可用510.44MiB、未使用
reserved927.25MiB，都无法满足9.27GiB请求，不能只归因于碎片或用退出143替代
CUDA异常证据。未改变entropy实现、精度或重计算配方来强行提高MBS。

原始manifest、registry提交、terminal sacct、完整错误日志及SHA封存于主项目
`runtime/mor-pretraining/finaldata-driver-v2/evidence/7215099/`。原driver停止后，
仅在显式 `--reconcile-rejected 7215099` 重审已完成档位、拒绝记录和实际终态后
恢复；原失败状态归档 `failed-7215099.json`，没有手改成功标记或重提失败作业。
独立政策审计 `select_mbs=2, next_candidate=None`：B组扫描完成，选MBS2，
吞吐364,174.45 tokens/s、70.73890%整卡峰值，不继续MBS8。随后推进C/MBS1。
生产源码58b909e3...及bundle08901c99...不变，四组正式验收凭据仍未签发。
C/MBS1已实际提交 `7215250`，输出 `runtime/mor-pretraining/finaldata-c64-mbs1-v1`；
继续相同64GPU/EP16/segment4、最终数据和20+30步调优合同，不复用B的MBS选择。

### 2026-09-17 C/MBS1最终数据短跑通过

`7215250` COMPLETED、退出0:0，耗时20m04s。driver及独立审计核验50次更新、
初末全量验证、step50 checkpoint和64rank显存证据；50个全局输入hash与
A/MBS2、B/MBS1、B/MBS2逐步一致。仍为相同生产源码58b909e3...、最终数据
fbbeba80...、候选环境487e6368...及64GPU/EP16/segment4/GBS2048/seq4096。

后30步吞吐524,385.920 tokens/s；整卡峰值57,916,456,960 bytes（29.26587%），
allocated/reserved峰值43,976,747,520 / 46,762,295,296 bytes，每rank至少
53,189次20ms采样。应用耗时1096.5946秒、19.49501 GPU-hours。
初始验证NLL=12.38381874，最终NLL=7.45157662、PPL=1722.57686，
第50步训练LM loss=7.44973780。只用于MBS调优，不计正式tokens，不作模型效果结论。
当前峰值低于80%，继续C/MBS2，不能仅凭MBS1成功冻结C。
证据：主项目 `runtime/mor-pretraining/finaldata-driver-v2/evidence/7215250/`。
下一档C/MBS2已提交 `7215526`，输出 `runtime/mor-pretraining/finaldata-c64-mbs2-v1`，
继续相同50步调优与完整验证；此处仅记录提交，未计为通过。

### 2026-09-17 C/MBS2最终数据短跑通过

`7215526` COMPLETED、退出0:0，耗时15m55s。driver与独立审计通过50次更新、
初末全量验证、step50 checkpoint、64rank显存，以及C/MBS1与MBS2的合同和50步
全局输入hash一致性。仍为相同源码58b909e3...、数据fbbeba80...、环境487e6368...。
后30步吞吐743,394.095 tokens/s；整卡峰值90,759,954,432 bytes（45.86208%），
allocated/reserved峰值74,150,006,784 / 79,532,392,448 bytes，每rank至少40,947次
20ms采样。应用计时847.2836秒、15.06282 GPU-hours。最终验证NLL=7.39699742、
PPL=1631.07962，第50步训练LM loss=7.39821041。仍是调优结果，正式tokens为0。
显存低于80%，继续C/MBS4；MBS2仅为当前最优候选，尚不能冻结C。
证据：主项目 `runtime/mor-pretraining/finaldata-driver-v2/evidence/7215526/`。
下一档C/MBS4已实际提交 `7215961`，输出 `runtime/mor-pretraining/finaldata-c64-mbs4-v1`；
保留相同配置和50步调优要求，此处不代表该档通过。

### 2026-09-17 C/MBS4通过及C组选择

`7215961` COMPLETED、退出0:0，耗时13m20s。driver和独立三档联合审计通过
全部50次更新、相同全局输入、初末完整验证、step50 checkpoint及64rank显存。
源码58b909e3...、数据fbbeba80...、环境487e6368...和科学配置保持不变。
后30步吞吐1,081,407.559 tokens/s，比C/MBS2高45.47%；整卡峰值160,857,260,032
bytes（81.28307%），allocated/reserved峰值137,853,065,216 / 149,703,098,368
bytes，每rank至少31,869次20ms采样。应用计时661.2036秒、11.75473 GPU-hours。
最终验证NLL=7.47700590、PPL=1766.94245，第50步训练LM loss=7.37920226。
MBS按预定显存与吞吐规则选择，不根据这些独立短跑的早期loss挑选。

峰值<=90%且各项检查通过，MBS4合格；峰值>80%，不继续MBS8。
独立审计 `sweep_complete=true, selected_mbs=4, next_mbs=null`，C组扫描完成。
证据在主项目 `runtime/mor-pretraining/finaldata-driver-v2/evidence/7215961/`。
当前A/B/C分别选MBS2/2/4，driver继续D/MBS1；正式凭据尚未签发。

D/MBS1已提交 `7216174`（registry `20260917-054130-45de`），实际SLURM状态
RUNNING；输出 `runtime/mor-pretraining/finaldata-d64-mbs1-v1`，仍为64GPU、
相同最终数据与冻结源码的50步扫描，不是正式训练，也不是把A/B的已选MBS退回1。
各组需要独立测量动态路由/激活内存和吞吐，其他组MBS2通过不能替代D的测量。
该任务初始完整验证NLL=12.32989978、有效targets=50,753,524，严格因果阈值
评估实测平均递归深度2.30299564。首个更新完成，LM loss=12.32690076、
depth auxiliary=0.00225909、grad norm=5.89289999，均有限；首步含初始化耗时
49.05秒，不作为稳定吞吐。仍需完成20步warmup和30步测量及最终验证/保存后
才能判断此档合格，不能据首步宣称扫描完成。

### 2026-09-17 D/MBS1最终数据短跑通过

`7216174` COMPLETED、退出0:0，耗时43m12s。driver与独立审计均通过完整50次更新、
初末全量严格因果验证、step50 checkpoint及64rank显存证据。全部50步的输入hash、
token游标、学习率与A/MBS2、B/MBS2、C/MBS4逐步一致。源码58b909e3...、数据
fbbeba80...、环境487e6368...，64GPU/EP16/segment4/seq4096/GBS2048保持不变。

后30步吞吐196,839.685 tokens/s；整卡峰值71,902,363,648 bytes（36.33311%），
allocated/reserved峰值57,939,423,232 / 60,672,704,512 bytes，每rank至少120,107次
20ms采样。应用计时2487.1776秒、44.21649 GPU-hours。第50步训练LM loss=7.19829341；
最终严格因果验证NLL=7.18205319、PPL=1315.60668，有效targets=50,753,524，实测
平均递归深度1.60861995（初始2.30299564），不把训练容量当作评估深度。

扫描中的慢步保留在后30步总耗时中，没有挑选最快步。读取状态时曾遇到一次SSH
连接断开，重试同一job后确认更新继续；没有因此重启或重复提交训练。
此档峰值低于80%，独立审计要求继续MBS2；不能冻结D。所有短跑正式tokens仍为0，
上述早期loss不作为模型优劣结论。证据：主项目
`runtime/mor-pretraining/finaldata-driver-v2/evidence/7216174/`。
下一档D/MBS2已提交 `7217234`（registry `20260917-062900-d237`），输出
`runtime/mor-pretraining/finaldata-d64-mbs2-v1`；提交后实际状态PENDING，未计为通过。

### 2026-09-17 D/MBS2最终数据短跑通过

`7217234` COMPLETED、退出0:0，耗时26m42s。driver与独立两档联合审计通过完整
50次更新、相同合同（仅MBS不同）、相同50步全局输入、初末验证、step50 checkpoint
与64rank显存。仍为source58b909e3...、数据fbbeba80...和环境487e6368...。
后30步吞吐376,492.297 tokens/s；整卡峰值114,438,897,664 bytes（57.82732%），
allocated/reserved峰值98,793,687,040 / 102,921,928,704 bytes，每rank至少71,896次
20ms采样。应用计时1483.1363秒、26.36687 GPU-hours。
第50步训练LM loss=7.18690529；最终严格因果验证NLL=7.14636891、PPL=1269.48795，
实测平均递归深度1.31824796。初始验证NLL和深度与D/MBS1完全相同。

峰值低于80%，必须继续MBS4；独立审计`best_so_far=2, next_mbs=4,
sweep_complete=false`。不按早期loss挑MBS，不把短跑记入正式训练tokens。
证据：主项目`runtime/mor-pretraining/finaldata-driver-v2/evidence/7217234/`。
下一档D/MBS4已提交`7218675`（registry `20260917-071830-80d3`），输出
`runtime/mor-pretraining/finaldata-d64-mbs4-v1`；实际状态PENDING，尚未计为通过。

### 2026-09-17 D/MBS4失败证据与四组扫描完成

`7218675`最终FAILED、143:0、9m37s；完成12次更新，第13次更新forward在rank9
发生`torch.OutOfMemoryError`。位置仍为native `linear_cross_entropy.py:33`，
`(exp_logits * logits).sum(dim=-1)`申请9.27GiB的完整词表FP32临时张量。
该rank当时free=7.90GiB、allocated=155.01GiB、reserved但未使用=10.97GiB，
总容量184.31GiB。这次free与未用cache之和大于申请量，不能像A/B的部分rank
那样仅凭这些数字排除碎片/不可回收分配影响；只能确定当前固定实现/allocator下
MBS4未完成必需的稳定性短跑，故不合格。不改allocator、精度或模型配方来强行通过。

`record_rejected_trial.py`封存真实registry、manifest、terminal sacct和异常日志；
日志SHA256=`316591e0d69daf34df7629a1c9c9c7deec640669367246fee2f6030e978a6311`。
driver遇到失败后退出，显式`--reconcile-rejected 7218675`重审后归档失败状态并
正常完成整个扫描。独立三档联合审计D得到`selected_mbs=2, next_candidate=None`，
没有将12步当成50步通过，也没有补造峰值/吞吐。四组最终选择为A/B/C/D=`2/2/4/2`；
GBS仍2048，64GPU下梯度累积分别16/16/8/16。此结论仅表示MBS扫描完成，尚未
签发正式验收凭据或启动正式训练。
本次D/MBS4包括失败封存与12步原始记录的71个文件已复制到远端验收输入目录，
逐文件SHA256与本地一致；finaldata证据现659个文件，screening仍144个文件。

### 2026-09-17 已完成扫描证据的远端验收副本

为后续在集群CPU端运行issuer，将已审计JSON/日志复制到
`runtime/mor-pretraining/acceptance-inputs-source58-v1/`：`finaldata/`含A/MBS2、
A/MBS4拒绝记录、B/MBS1/2/4拒绝记录、C/MBS1/2；`screening/`含pilot A/MBS1/2。
复制不包含模型权重，不删除或修改原始产物。复制后逐文件SHA256比较：finaldata
372个文件、screening144个文件，两端全部一致。随后追加C/MBS4 `7215961/`
的72个文件，并逐文件复核SHA256，全部一致；finaldata共444个文件。
随后D/MBS1 `7216174/`的72个文件也已追加、两端逐文件SHA256全部一致，
finaldata共516个文件。随后追加D/MBS2 `7217234/`的72个文件，两端逐文件SHA256
全部一致；finaldata现共588个文件，另有screening144个文件。后续D新结果仍须追加并复核；
这个目录只是待验收证据副本，不是正式接受凭据，尚未执行issuer或启动正式训练。
正式campaign所需的最终数据manifest另已取回主项目
`runtime/mor-pretraining/evidence/data-50b-manifest.json`，SHA256为
`fbbeba8051302c86818f4edd8d95131fb61a579bbbfa5f0ed5d54def7e7d6153`，
与所有finaldata扫描合同一致。由其训练token数重新计算得到5954次更新、
49,945,772,032个消费token与3,469,450个舍弃尾部token；不含验证token。

### 2026-09-17 验收作业的显式提交入口

新增独立 `scripts/pretraining/launch_probes.py`，为现有 `probe_suite.py` 生成
cluster-run作业，默认只render，显式 `--submit` 才同步/提交。tiny smoke、auxiliary
和shared检查固定32ranks；完整初始化、indexed恢复固定64ranks，均segment4。
indexed模式强制 `--full-width` 并传入所选MBS；拒绝缺fixture/data、重复arm或
错误world size，不降低已有验收条件。smoke提交时获取真实Slurm topology。
复用生产content-addressed source/config bundle，并核对native commit及干净状态；
生产源码仍58b909e3...，bundle仍08901c99...。此入口不签发凭据、不自动启动训练。

A完整宽度/MBS2恢复脚本已dry-run，核对16nodes×4GPU、`#SBATCH --segment=4`、
最终数据、官方tokenizer、候选venv以及固定bundle均正确。尚未提交这项GPU作业；
待当前顺序扫描释放资源后执行，不能把render或CPU测试当作GPU恢复通过。
本次probe guard回归44项通过，新增7项覆盖提交入口的scope和参数传递。
随后完整CPU回归190 passed、2 skipped（54.68s）；跳过的两项要求native/CUDA，
没有计入GPU验收。相关独立脚本及测试ruff检查通过。
另外已完成本版smoke、auxiliary、shared、全宽initialization以及A/B MBS2、
C MBS4全宽indexed恢复入口的render核验：固定source bundle、segment4和32/64rank
范围一致，恢复入口带 `--full-width`；均未因此计为GPU通过。A/B恢复共用一个
顺序suite、C单独suite，时间上限各1小时，输出到新的gates-source58目录。

### 2026-09-17 当前冻结版本tiny验收实际提交

D/MBS2 `7217234`处于资源排队时，提前提交与所选MBS无关的32GPU tiny语义/因果
验收，减少后续串行等待：`7217539`（registry `20260917-063517-4651`），通过
`launch_probes.py --kind smoke --arms A B C D --world-size 32 --submit`，输出
`runtime/mor-pretraining/gates-source58/smoke32-v1/`。suite内部四组按顺序启动独立
进程，固定source58b909e3...、bundle08901c99...及候选环境；使用segment4和真实
Slurm topology。不改变D扫描配置、不与扫描共用GPU、不计入正式tokens。
这是必需的当前版本前置验收，提交本身不代表通过；后续仍需auxiliary、shared、
全宽初始化、所选MBS外部indexed恢复和issuer验证。先前“待扫描全部结束才启动
验收”的操作顺序在此调整；正式训练仍必须等待四组扫描和全部验收均完成。
提交后已确认RUNNING，节点`nvl72091-T[01-08]`；D/MBS2此时仍PENDING。

该任务最终COMPLETED、退出0:0、耗时4m48s。取回四组全部32rank JSON报告后，
`inspect_smokes.py`联合审计通过：当前source58b909e3...一致、optimizer unique
coverage通过、dense/expert-DP初始化bitwise一致、独立expert初始化各不相同，
B/C与B/D的初始backbone逐参数bitwise一致。tiny独立参数量A=154,221,696，
B/C=64,297,344，D=64,297,728（depth router额外384）。四组每rank八个逐前缀
因果预测比较的实测最大绝对误差均0；D覆盖178个zero-active rank/round，未死锁。
此结论仅适用于tiny语义检查，不代替完整30B宽度训练/恢复验收。证据本地副本：
主项目 `runtime/mor-pretraining/evidence/gates-source58-smoke32-v1/`；集群issuer
仍读取原始 `gates-source58/smoke32-v1/{a,b,c,d}`。

接着提交四组auxiliary suite `7217688`（registry `20260917-064316-9a7f`），32GPU、
segment4，fixture为本版已通过smoke的B/config.json，输出
`runtime/mor-pretraining/gates-source58/auxiliary32-v1/`。提交记录不代表梯度及归一化
检查通过；须待全部32rank报告审计后再计入issuer输入。
排队期间仅将该辅助作业的Slurm TimeLimit从30分钟调整为15分钟，`scontrol show job`
已确认；生成时的原sbatch仍保留30分钟记录。依据是历史相同tiny检查A/B/C加D约
5分钟，给出额外余量以便backfill。只调整这个明确job的调度时限，不改源码、数据、
验收范围或D/MBS2的一小时时限；若达到TIMEOUT应诊断并按失败处理，不能计为通过。

共享梯度检查不依赖auxiliary结果，fixture已由本版smoke验证，因此另提交32GPU
`7217889`（registry `20260917-065221-c004`），`--kind shared --arms B`，输出
`runtime/mor-pretraining/gates-source58/shared32-v1/b/`，时限10分钟（历史同类2m07s）。
同步遇到一次临时传输失败，由既有pre-submit重试处理；只产生一次SLURM提交。
本版检查仍是tied参数梯度对比同native kernel的显式untied展开求和，不把它表述为
独立Transformer kernel reference。提交不是通过，须审计所有32rank报告。
同期已确认D/MBS2 `7217234`与auxiliary `7217688`实际RUNNING且分配不同节点；
这些前置验收不改变四组正式训练的顺序，也不计入正式训练tokens。

`7217889`最终COMPLETED、退出0:0、耗时2m07s。全部32rank报告联合审计通过，
共14,816项参数梯度检查，最小tensor similarity=0.9999907854166675，最小cosine=
0.9999919667793155，最大LM loss差=0。全部报告绑定source58b909e3...，probe SHA256
`386f1a18f271f3947c783ecebb580f0c24336d1909b54cc79da3d103e25ecd00`。
本地证据在主项目`runtime/mor-pretraining/evidence/gates-source58-shared32-v1/b/`；
issuer使用集群原始`gates-source58/shared32-v1/b/`。这是共享梯度链式法则验收，
不替代独立kernel精度参考。

### 2026-09-17 auxiliary suite首次运行的进程崩溃诊断

`7217688`最终FAILED、退出143:0、运行2m32s，不是15分钟时限触发，也不是数值
断言失败。A已写出结果，suite启动B子进程时rank10（`nvl72145-T03`）首先记录
`SIGSEGV:11`，Slurm随后取消整个step。取消后rank4/5/6/7记录
`ModuleNotFoundError: megatron.core.transformer.mla_qk_norm_config`，并出现
`os.getcwd`相关FileNotFoundError及TCPStore关闭错误；仅凭日志顺序还不能证明
SIGSEGV与后续读取错误的因果关系。原始日志保留在主项目
`runtime/mor-pretraining/evidence/gate-auxiliary-7217688.log`，未计此suite为通过。

只读复核：该native模块两端SHA256均
`9be9beecdc16a270a2b4836ca2e3759796532fdc2f31d6318c311b1eb26d93b3`，远端ctime
仍为9月16日，不支持“本次同步改写该.py文件”的假设；整个native的772个Python
文件SHA256仍为`6e235d0f9097fb8b12e794f6ef0ac5e86c72747df2275c3043c9471a3449d5d3`。
节点随后状态IDLE+PLANNED，不能据此断定硬件健康或确认根因；项目根未找到core文件。

为复现补充诊断，`launch_probes.py`仅新增`PYTHONFAULTHANDLER=1`，宿主和容器
payload均有export；不改生产source/config bundle、不修改任何数学断言或容差，
不改变正在运行作业的环境。probe回归44项通过，ruff通过，render仍绑定bundle
08901c99...。在新的`gates-source58/auxiliary32-v2`目录重跑同一四组suite并保留
原失败结果；当前只做了诊断增强，尚未确认SIGSEGV根因或宣称修复。
带诊断重跑实际任务为`7218084`（registry `20260917-065932-638b`），32GPU、
时限15分钟，四组与断言不变；提交时env记录含`PYTHONFAULTHANDLER=1`。

该重跑最终FAILED、143:0、运行3m30s，A/B/C写出通过报告，但D启动时rank0在
`nvl72033-T12`发生SIGSEGV。Python fault handler给出主线程当时位于动态生成的
`ImplConfig.__init__`，调用来自native `_build_impl_cfg` / `Runtime.build_model`，
尚未进入auxiliary数值断言。fault handler显示的是`Thread`而非可明确归因的原生
faulting-thread回溯，不能据此证明dataclass本身有bug。原始日志：主项目
`runtime/mor-pretraining/evidence/gate-auxiliary-7218084.log`。两次均未计整套验收通过。

旧节点core查询被`pam_slurm_adopt`拒绝（已无该节点allocation），未绕过权限。
只读查看固定sqsh确认存在`/usr/local/cuda-13.2/bin/cuda-gdb`，因此新增独立probe
的显式`--debugger`选项：suite用batch debugger执行相同子命令，并在异常时输出
`thread apply all bt 40`；必须使用`--return-child-result`且仍`check=True`，避免把
调试器正常退出误当子程序通过。默认路径完全不包装debugger，禁用debuginfod联网
取符号；不改生产source/bundle或probe的数学检查。45项probe回归与ruff通过，
渲染核对32GPU、segment4、固定bundle、fault handler和debugger参数均正确。
这是原生崩溃定位入口，不是根因修复或正式训练验收结论。
原生栈诊断已提交`7218580`（registry `20260917-071455-3044`），32GPU、30分钟，
输出`gates-source58/auxiliary32-native-stack-v3/`，保持相同四组及数值断言。
该任务只用于捕获原生faulting-thread，不把“在debugger下未崩溃”当作原问题修复。

该任务最终COMPLETED、0:0、7m04s；全部128份rank报告已审计，四组source和
auxiliary probe指纹与当前版本一致，optimizer全局分片覆盖通过。A/B/C/D的最大
FP64归约参考relative-L2分别为6.47022e-9、5.76745e-9、5.44923e-9、4.79693e-9。
日志有128次子进程正常退出，没有SIGSEGV，因此没有取得原生崩溃栈，不能宣称
修复。cuda-gdb期间的CUDA_ERROR_INVALID_CONTEXT诊断也不能直接归为原崩溃根因。

为减少调试器与正常启动的环境差别，后续诊断包装显式设置
`set disable-randomization off`并打印设置；gdb默认尝试关闭ASLR，旧日志没有
对应权限警告，但这不足以证明ASLR就是根因。此改动仅在显式`--debugger`时生效，
默认运行路径、生产模型、auxiliary数学断言和退出状态检查不变。下一次仍是故障
定位，不以调试器下的成功替代正常模式稳定性检查。
保留ASLR的诊断作业为`7219199`（registry `20260917-073534-0649`），32GPU、
15分钟，输出`gates-source58/auxiliary32-native-aslr-v4/`；提交前一次rsync连接失败
经已有有界重试恢复，只产生一个SLURM作业。45项probe回归及ruff通过。

### 2026-09-17 当前版本完整宽度初始化验收提交

`7219258`（registry `20260917-073751-de53`）提交A/B/C/D顺序初始化检查，64GPU、
EP16、segment4、15分钟预约，输出`gates-source58/initialization64-v1/`。
使用最终50B数据manifest、官方Qwen config和当前冻结source58b909e3...；不用
debugger，不执行训练更新。此检查不依赖D的MBS选择，故在D/MBS4排队期间提前
提交；与其他作业使用独立allocation，不改变正式四组顺序。须审计全部256份
rank参数指纹后才计为通过，提交本身不代表完成。

`7219258`最终COMPLETED、0:0、2m57s，全部256份rank指纹经
`inspect_smokes.py --initialization-run ... --world-size 64`联合审计通过：
dense和expert-DP副本bitwise一致、独立expert初值不同、B/C及B/D的backbone
逐参数bitwise一致。A参数30,532,122,624，B/C为13,084,744,704，D为
13,084,750,848。source58b909e3...及官方config指纹均一致；本地证据目录为
`runtime/mor-pretraining/evidence/gates-source58-initialization64-v1/`。

A/B完整宽度MBS2的真实indexed外部恢复检查已提交`7219391`（registry
`20260917-074429-f383`），64GPU、segment4、1小时，输出
`gates-source58/indexed64-ab-mbs2-v1/`。每组原进程两步、独立进程恢复并比较下一步；
不计正式tokens。这一作业仍使用原启动环境，未热改已提交脚本。

作业继续B阶段时，已取回并审计A的全部64份`a-resume/resume-rank-*.json`：
full-width/MBS2、source/data/config合同一致，model/master/Adam/RNG恢复及下一更新
与不中断参考bitwise通过；游标16,777,216，下一输入hash为
`f0168b3c28e3d01238446ab31984c3fc7ed225442c7d436609cfacbe75f7a63e`，
与A最终数据50步调优的第2步一致。只计A已完成，不提前把整个A/B作业记为通过。

`7219391`最终COMPLETED、0:0、11m23s。A/B全部128份外部恢复报告已联合审计，
两组均为完整宽度/MBS2、相同最终数据及5954步日程，model/master/Adam/RNG、
optimizer学习率、下一更新和游标bitwise通过；B的下一全局输入hash也与调优第2步
及A完全一致。证据本地副本为`evidence/gates-source58-indexed64-ab-mbs2-v1/`。

### 2026-09-17 auxiliary崩溃原生栈与启动环境竞争

保留ASLR的`7219199`在5m00s后FAILED、143:0；rank7的原生faulting thread是
`pt_nccl_heartbt`，调用链为libc `getenv` → `c10::utils::get_env` →
`c10d::getCvarString` → `c10d::DumpPipe::DumpPipe(int)` →
`ProcessGroupNCCL::HeartbeatMonitor::runLoop`。原日志完整保存在
`runtime/mor-pretraining/evidence/gate-auxiliary-7219199.log`。这证明不是先前
Python栈中的dataclass直接触发数值异常，也不是MoE/depth辅助损失断言失败。

固定native的`Runtime.build_model`先`init_process_group`，再在
`_apply_attention_backend_env`中写入`NVTE_FLASH_ATTN/FUSED_ATTN/UNFUSED_ATTN`；
它们未在原probe启动环境中设置。该调用链与PyTorch上游报告的环境表重分配竞争
吻合：[issue 175724](https://github.com/pytorch/pytorch/issues/175724)，
[PR 167523](https://github.com/pytorch/pytorch/pull/167523)。上游报告不是本地修复
证明；本地目前抓到getenv崩溃及晚写环境变量的代码，未直接捕获具体realloc指令。

独立probe启动器现提前export相同的fused选择值`0/1/0`，在Python/NCCL线程启动前
建立这些key，避免该native后续设置首次插入key引起环境表扩容。与现有
`ATTENTION_BACKEND`核对，不改变attention算法、不关闭watchdog/heartbeat、不放宽
数值检查。生产source/bundle和auxiliary数学脚本仍不变；下一步不带debugger复验。
这尚不是已验证修复，也尚未处理正式启动入口的同类风险，不能据此放行正式训练。

不带debugger的四组复验已提交`7219512`（registry `20260917-075015-3af5`），
32GPU、15分钟，输出`gates-source58/auxiliary32-early-env-v5/`。registry env及
宿主/容器payload均明确包含三个预设值，数学probe SHA仍为13b57a83...。

`7219512`最终COMPLETED、0:0、3m49s，未使用debugger；全部128份rank报告审计
通过。A/B/C/D的最大FP64归约参考relative-L2分别为6.43697e-9、5.86498e-9、
5.41641e-9、4.21060e-9，精确optimizer分片覆盖与microbatch累积检查通过；源码
及数学probe指纹未变。本地证据目录`evidence/gates-source58-auxiliary32-early-env-v5/`。
这支持提前建立TE环境变量的启动修正，验收采用本次正常模式报告而不是debugger
下的偶然成功。仍不声称仅一次成功证明所有PyTorch启动竞争已被修复。

正式campaign新增显式`--python .../scripts/pretraining/fused_python.sh`启动入口：
在exec同一个`gb200-env-nvrx060/bin/python`前设置三个TE选择值，遇冲突值则报错，
原样传递argv及子进程退出码。wrapper SHA256为
`100fa4b2473e0383566eae279273fa261b4959f9a5a767fdfff086a767ea85d8`。
issuer把wrapper SHA写入真实凭据，campaign核对本地hash、所选启动路径和每次提交前
远端文件hash/可执行权限；不能仍用裸Python启动正式任务。现有数学源码、native、
环境依赖及所有验收条件不变，source仍58b909e3...，bundle仍08901c99...。
这是应用启动顺序的修正，不声称修补了PyTorch内部所有后台getenv风险；GPU正常
模式复验及使用该入口的完整宽度恢复尚待完成。相关77项CPU检查及新增campaign
wrapper约束后共87项通过（含pinned native selector检查），ruff和shell语法通过。

C完整宽度MBS4外部恢复检查已通过新wrapper显式提交`7219612`（registry
`20260917-075501-15c6`），64GPU、segment4、1小时，输出
`gates-source58/indexed64-c-mbs4-v1/`。新wrapper随probe脚本同步，registry command
和env都记录提前设置的TE选择值；这也是该启动入口的真实GPU验证，结果尚待审计。
D扫描完成后，完整宽度MBS2恢复检查也通过同一wrapper提交`7219752`（registry
`20260917-080017-6327`），64GPU、segment4、1小时，输出
`gates-source58/indexed64-d-mbs2-v1/`。C/D仍须实际完成并核对所有rank，不能把提交
当作恢复通过。wrapper远端可执行权限及SHA256已与本地核对一致。

在确认两作业仍PENDING后，依据A/B双组恢复11m23s的实测，将C `7219612`的
Slurm TimeLimit从1小时调整为20分钟，D `7219752`调整为30分钟，以便backfill。
`scontrol show job`已核实新时限及64GPU/segment4；原生成脚本仍保留1小时记录。
不改变模型、数据、MBS或检查范围；若超时仍按失败诊断，不把部分结果记为通过。
最终issuer的CPU-only任务已dry-run核对：1节点、4 CPU、0 GPU、20分钟，引用
四组完整验收及11项finaldata扫描/拒绝记录、2项pilot screening。当前仅准备了
调用，没有提前运行issuer或生成凭据，仍等待C/D完整恢复。
随后已确认两作业实际RUNNING。预训练相关CPU回归`tests/test_pretraining*.py`
共99项通过，显式提供pinned native目录，selector检查没有跳过。另只读核对当前
固定sqsh内TE的`dot_product_attention/utils.py:412–417`，后端选择函数读取三个
环境变量；提前设定的`0/1/0`与native原先build_model指定的fused选择相同，
未换用其他attention算法。该源码检查不代替实际GPU恢复报告。

### 2026-09-17 选定MBS的完整恢复验收收尾

C `7219612` 最终COMPLETED、0:0、5m37s；D `7219752` 最终COMPLETED、0:0、
5m56s。各64份resume-rank报告已下载并逐份审计：完整宽度、64GPU、C/MBS4、
D/MBS2、同一source58与最终50B数据、5954步完整日程。外部进程恢复后的模型、
FP32 master、Adam状态、RNG、optimizer LR及下一更新结果均与不中断参考bitwise
一致。两组游标均为16,777,216，第二批输入hash为`f0168b3c...a63e`，分别与
C `7215961`、D `7217234` 的调优第二批一致。结合此前A/B，选定MBS的四组
完整宽度外部恢复检查全部取得证据；调优和恢复探针均不计入正式训练token。

本地证据位于`runtime/mor-pretraining/evidence/gates-source58-indexed64-c-mbs4-v1/`
及`gates-source58-indexed64-d-mbs2-v1/`。这同时验证了新fused Python wrapper在
真实完整宽度模型上的启动与恢复路径，但不代表所有潜在运行故障已经排除。

### 2026-09-17 最终凭据汇总与正式作业时限

首次issuer提交（registry `20260917-081943-6156`）遗漏CPU分区，SLURM明确拒绝
在`batch`提交0GPU任务；无scheduler ID、未执行issuer。只读`sbatch --test-only`
重现该错误，修正为显式`--partition cpu`后提交 `7220068`（registry
`20260917-082158-6f1b`），1节点/4CPU/0GPU/20分钟，已确认RUNNING。此时尚未
生成凭据；不能把提交成功视为数据hash或验收已完成。

另核对集群当前`batch` MaxTime=4小时，而原实验recipe预约24小时。为合法使用
普通队列，新增运行产物`runtime/mor-pretraining/formal-batch-4h.yaml`，解析后
逐字段比较确认仅SLURM.time_limit改为4小时，四组launch_spec科学配置检查通过。
不修改集群默认、生产Python或冻结bundle。正式driver使用`--max-updates 400`：
按最慢入选档约364k tokens/s，400更新约2.56小时计算，另留数据审计、验证、
checkpoint及调度余量。首个1B阶段仍120更新；之后allocation之间恢复完整状态，
不重置学习率、不改变5954更新总量或四组1B→10B→全量的推进顺序。

`7220068`最终COMPLETED、0:0、5m47s，完整issuer通过后生成四组凭据于
`runtime/mor-pretraining/acceptance-source58-final-v1/`，选定MBS为A2/B2/C4/D2。
本地副本`runtime/mor-pretraining/evidence/acceptance-source58-final-v1/`逐文件SHA
已与远端核对：A=`8d9cf583...ddfa`、B=`04321bbf...3100`、C=`13845252...5fba`、
D=`b7a1a2c5...c294`。这不是手工填写的checks：issuer实际核验四组语义smoke、
完整宽度初始化、正常模式auxiliary、共享梯度、选定MBS完整恢复、完整扫描和
各数据文件hash后才生成。CPU预训练回归再次99 passed（4.81s）。

正式campaign只读预览通过：第一项A，从0到120步、1,006,632,960输入tokens；
四组MBS匹配凭据。开始执行driver，状态目录`runtime/mor-pretraining/formal-driver-v1/`，
使用上面的4小时recipe、每段至多400更新及fused wrapper。此时仅启动driver，
尚不能把后续提交或排队视为已完成首个训练更新。

首个正式A作业已提交`7220271`（registry `20260917-083131-c6ae`）：
`mode=train`、MBS2、64GPU/16节点、EP16、segment4、无resume参数、stop_tokens
1,006,632,960。命令与验收A凭据及冻结源码绑定。首段仅120更新，确认PENDING后
将本作业TimeLimit由4小时缩为2小时；`scontrol show job`核对成功，后续每段400
更新仍用4小时recipe。原脚本保留4小时、Slurm记录2小时，不隐去运行时调整。
输出目录为`runtime/mor-pretraining/formal-source58-v1-a-000000-000120/`，日志
`runtime/slurm/logs/mor-a-mbs2_7220271.log`。此记录时仍PENDING，尚未计为训练启动成功。

排队期间调度器曾给出预计开始时间集群当地13:50（会随队列变化）。A/MBS2最终
数据50步实测28分钟；外推120步计算及相同固定启动/验证开销约55分钟。再次确认
`7220271`仍PENDING后，将这一次120步预约从2小时缩为90分钟以增加backfill机会，
保留约35分钟估算余量；不改变stop_tokens，不修改后续400步/4小时设置。调度器
预计时间不视为承诺，只有实际训练更新才计为启动成功。

### 2026-09-17 正式A训练启动成功

`7220271`于21:51:20 UTC实际开始运行，命令明确为`mode=train`、MBS2、64GPU。
已下载并审计前8步真实更新，保存于主项目
`runtime/mor-pretraining/formal-driver-v1/startup-evidence/7220271/`。
manifest的experiment/source/data/environment/model-config与签发A凭据逐项一致；
总日程5954步、49,945,772,032 tokens，独立参数30,532,122,624。
重新调用生产`validate_ep_locality`检查全部64rank：四个EP16组分别满足单domain、
4完整节点、每节点4rank，没有以节点名称推断代替检查。

初始全量验证NLL=12.386161704331121、targets=50,753,524；首步训练LM loss=
12.384249215014279、grad_norm=5.093868732452393、LR=5.038629492777964e-06。
首两步输入hash匹配扫描记录；8步序号/token连续，loss/LR/梯度/耗时均有限。
第8步累计67,108,864输入tokens、LM loss=10.31216055387631。日志行在成功
optimizer更新后写入，因此不是仅凭RUNNING认定训练启动。driver仍顺序推进
A/B/C/D的1B→10B→全量；此处仅完成启动目标，不声称四组或1B阶段已完成。

此前全量数据A/MBS1作业7213156属于重复扫描，已在无完成更新时取消；当前正式
任务没有退回MBS1。最终A/B/C/D的MBS仍为2/2/4/2，GBS均2048。

### 2026-09-18 四组独立排队与阶段同步

用户明确批准由四组串行改为各64GPU独立排队/允许并发，仍保持EP16和
`--segment=4`（16节点，每4节点一个segment，每节点4GPU）。这不是组间训练依赖；
各组后续训练段只恢复自己的checkpoint。保留阶段同步：全部到120步才进入
1193步阶段，全部到1193步才进入完整5954步阶段。最大同时4作业/256GPU；
A已完成1B时不重跑A，B排队不阻止C/D提交。

独立脚本`drive_campaign.py`新增显式`--parallel-arms 4`，默认1仍为串行。
schema v2使用arm→active-job映射；PENDING也占该arm的槽位，但绝不能当作
已完成来跨阶段。每组每次最多400更新，完成后原审计检查仍全部执行，跨组
输入hash仍比较。任意失败/提交不明确停止新提交，不擅自取消其他已提交作业，
不自动重试sbatch。SSH状态读取错误按job分别计数，避免另一个正常job清零错误。

迁移在同一排他锁下进行：保留旧state备份，只允许调度选项变化，重新读取并审计
全部已完成产物，核对completed游标；保留既有scheduler ID。科学配置、凭据、
源码bundle和checkpoint恢复合同都不能随迁移变化。B `7237775`明确保留，
A `7220271`的120步产物已重新审计，不手改完成标记。

`--first-stage-time-limit 1:30:00`仅为各组从0开始的1B段生成运行时recipe，
只覆盖SLURM.time_limit，不改冻结配置；后续段继续原4小时recipe。依据是A实际
55m25s、B/D选定MBS的50步短跑约27分钟，90分钟留出验证/保存余量。
已有B在PENDING时通过`scontrol update`改为90分钟，作业ID/SubmitTime不变；
不取消重排。时间估算不保证避免TIMEOUT，发生时仍按失败处理。

排队快照（2026-09-18约01:54 UTC）：B normal优先级122810；运行中的
`hero-n3`例7241199为1092523，`hero-res`例7240118为1092150。另有normal
作业7236501/7236502，各128节点（512GPU），优先级142714。节点状态快照含
1905个alloc（含alloc$）、341个resv、347个plnd及19个idle；这些状态会快速变化，
不能将idle/plnd总数直接视为满足4节点NVL segment和预约时长的可用资源。
B的Dependency为空、Reason=None，证明无Slurm依赖，但不能据此把等待唯一归因
于某一作业；高优先级竞争、预留/计划资源与满足布局的时间窗口共同影响调度。

新增并行调度、阶段屏障、乱序完成、同arm恢复、迁移保留作业/证据、非法游标与
未审计进度拒绝测试，以及真实事件循环的隔离提交测试；预训练CPU回归110项通过，
ruff通过。生产Python SHA仍
`58b909e3...359a`、bundle仍`08901c99...0b69`；未改变训练或验收数学代码。

B于2026-09-18 02:01:08 UTC由Backfill实际调起，排队3h11m54s；此时90分钟
时限已生效，scheduler ID/SubmitTime未变。时间先后与回填机制相符，但不把一次
启动证明为单独的因果实验。实际分配64GPU、16节点、SegmentSize=4，无依赖。
C已独立提交7241927、MBS4、64GPU/segment4/90分钟，无Slurm依赖。
D也已独立提交7242004、MBS2、相同64GPU/segment4/90分钟。02:05 UTC三组
registry命令与实际SBATCH逐项核验通过：16nodes、4GPU/node、segment4、无依赖，
MBS仍B2/C4/D2、GBS2048，0→120步。此时B RUNNING（尚无训练更新记录）、C/D
PENDING；不能把提交当作已开始训练或已完成。宿主后台控制进程与临时命令会话
分离，日志`formal-driver-v1/driver-parallel-host-v2.log`，状态仍同一目录。

### 2026-09-18 A/C早期NLL差异的只读核验

针对“同为普通Transformer，为何20层C优于48层A”，重读正式作业7220271、
7241927的全部120条更新及完整验证记录：每一步的输入hash、tokens、LR均一致；
验证targets均50,753,524，数据/源码/环境/model-config指纹一致。A/C实际独立
参数分别30,532,122,624和13,084,744,704，均调用`run_fixed`且不含块外递归gate。
初始验证NLL分别12.38616170/12.38381874；120步为6.36983693/6.08752734，
C−A=−0.28230960。末步训练LM loss分别6.34340087/6.08216352，差异不只出现在
验证集。训练step20 A/C=8.62324467/8.66790469，step40=7.80895813/7.69080409；
从step26起C在逐步同输入训练loss中持续领先。该记录不支持直接解释成“A过拟合”。

正式A/C的MBS为2/4；原生MoE load-balancing统计以rank/microbatch为单位，
所以GBS一致不能证明训练轨迹与MBS无关。另复核已有最终数据调优作业
A/MBS2 7213308与C/MBS2 7215526：相同源码、数据、环境、全部50步输入和LR，
除arm外experiment字段一致；50步验证NLL为7.58769781/7.39699742，C−A=
−0.19070039。这说明浅层早期领先并非仅由正式MBS不同造成；这仍是单seed短跑，
不能冒充120步同MBS实验或完整训练结论，也不能由它量化MBS对正式差距的贡献。

当前只确定统一配方下的早期优化表现不同，尚未定位具体因果机制。日程以5954步
为总长，warmup约59.54步，120步仅刚过约60个非warmup更新。初始化入口对所有
矩阵使用std=0.02，无额外按深度缩放；这为后续残差/逐层梯度及更新幅度诊断提供
线索，不足以认定初始化错误或梯度消失。需要进一步受控探针/消融才可归因。
本轮仅检查数据与代码，未更改正式训练、初始化、MBS、LR或提交新GPU实验。

### 2026-09-18 正式训练配额故障与恢复预检查

正式第二段120→520中，A `7243509`（1h09m50s）与B `7243525`
（1h04m16s）失败，原始异常为`[Errno 122] Disk quota exceeded`。
A在保存第300步时的PyTorch序列化写入失败，随后出现`unexpected pos`；
B在写`input-rank-*.jsonl`时失败。不能把后续退出143归因为OOM或NCCL。
执行器按既有合同停止新提交，没有取消独立的C/D。
C `7243558`已COMPLETED，121–520全部400次更新通过`audit_run`；
step520消费4,362,076,160 tokens，验证NLL=3.53748245、PPL=34.38025619。
D `7243582`在本轮15:05 UTC附近仍RUNNING，已到361步；这不是最终完成记录。

当前目录继承Lustre项目ID10089。项目块配额为268,435,456,000 KiB（250TiB），
只读快照已用266,280,390,688 KiB，余约2.01TiB。个人配额仅用约8.46TiB/100TiB，
group无块配额上限，整个Lustre仍有约2.7PiB可用；因此不能把这次故障描述为
个人100TiB用满或整个文件系统无空间。失败发生时的精确项目占用未留快照，
不能由稍后尚有空余推断当时写入正常。共享项目其他作业会改变可用配额。
本实验`runtime/mor-pretraining`占约7.5TiB，含调优、验收和正式训练权重。
A单checkpoint约399GiB，B/C/D各约171GiB；无保留策略的每100步保存不可持续。
清理需要明确授权，当前尚未删除权重、数据或他人文件，也未扩大或绕过项目配额。
只读清理候选核验：`pilot-fix1-a64-mbs{1,2}-v1`、
`finaldata-a64-mbs2-v1`、`finaldata-b64-mbs{1,2}-v1`、
`finaldata-c64-mbs{1,2,4}-v1`、`finaldata-d64-mbs{1,2}-v1`共10个已完成短跑，
均有完整step50标记，合计160个`.distcp`权重文件、2,565,126,228,958 bytes
（约2.33TiB）。该统计只是候选，不代表已删除；保留它们的JSON/日志和小型元数据
能维持已签发证据的复核，但删除权重后不能再直接恢复这些短跑的模型。

新增独立`inspect_recovery.py`只做受信任的本实验checkpoint只读预检查：
合同/游标、完整保存标记、已保存前缀的连续更新/LR/有限值/64rank输入日志、
必须存在的里程碑验证、DCP元数据引用范围及所有RNG文件CRC和两端SHA。
A/B第200步均通过：各16个native tensor shard，分别427,529,457,929和
183,219,663,613 bytes，64份RNG；各接纳121–200共80步前缀。
A第300步没有`pretraining-state.json`，明确拒绝作为恢复点。
预检查**没有读取全部tensor内容或完成GPU native恢复**，报告明确把这两项标为false；
后续实际恢复必须加载model/master/Adam/RNG，不能将预检查冒充恢复成功。

新增`reconcile_campaign.py`默认只读，只有显式`--apply`才原子更新状态并归档旧状态。
它重新查询SLURM实际终态、核对registry提交参数和签发凭据、检查远端分片仍与
预检查一致，重审旧历史、C完整产物、A/B保存前缀及跨组输入hash。
失败作业以独立`recovery`记录保留原计划、退出状态、报告SHA和实际恢复步；
不补造不存在的`complete.jsonl`、最终验证或显存报告，不接受活跃作业恢复请求。
旧失败日志保留，checkpoint之后的更新是被丢弃的计算，不重复计入有效训练tokens。
`audit_history`新增显式恢复记录审计；旧完整作业仍使用原`audit_run`。
试运行得到A200/B200/C520，D仍为原活跃作业；尚未应用状态或重启controller，
仍须先解决存储持续性。CPU回归120 passed，ruff通过。生产source58b909e3...、
bundle08901c99...、选定MBS2/2/4/2、64GPU/EP16/segment4均不变。

保留策略的只读入口`plan_checkpoint_retention.py`已实现，不包含删除功能。
仅扫描state中显式列出的正式作业目录，拒绝symlink/越界文件，核对marker合同和游标；
保留所有科学里程碑、每组最新两个有完成标记的checkpoint、活跃作业resume源和
全部不完整checkpoint，只列出旧`.distcp`分片的精确路径/size/inode/mtime。
这不是完整性证书：未读取全部native tensor，实际删除前必须另核验保留恢复点。
清单`formal-driver-v1/retention-candidates-v1.json`记录当前7个正式旧checkpoint
（A100、B100、C100/200/300/400、D100），合计1,526,847,544,547 bytes，约1.39TiB。
没有纳入A不完整300，也未删除日志、metadata或数据。清理权限仍待用户确认。
按A200→600、B200→600、C520→920及D剩余→520估算，若不清理，下一轮新增
checkpoint tensor约3.55TiB，高于当前约2.01TiB余量；尚不含其他用户增长与额外开销。
加入保留策略及越界/错误游标/活跃恢复源故障注入后，CPU预训练回归129 passed，
相关脚本ruff通过。没有修改生产训练快照或实际清理任何checkpoint。
D `7243582`随后到step400（3,355,443,200 tokens），写出该步
`pretraining-state.json`完成标记，并继续更新到402步；当前这次保存未因配额失败。
此处仅验证保存完成标记与继续训练，不冒充该checkpoint的外部进程恢复验收；
共享项目余量仍不足以无清理地推进四组下一轮，A/B仍未重启。

### 2026-09-18 D第二段完成与当前阻塞

持续监控同一作业`7243582`直到COMPLETED、0:0、2h41m47s；中间一次SSH
读取失败只重查原作业，没有重启或重复提交。D400的只读恢复预检查另已通过，
16个native分片共183,219,768,553 bytes、64份RNG、121–400共280步日志；
该前缀与C全局输入hash一致。仍不把预检查当成GPU外部恢复。

最后重新下载不可变完成产物，对C `7243558`和D `7243582`调用原`audit_run`：
各400次更新、正确5954步全日程、step520 checkpoint/游标、最终评估和64rank
显存证据通过；121–520全部400个全局输入hash一致。D step520为
4,362,076,160 tokens，验证NLL=3.4807452861966137、PPL=32.48392287581373，
严格因果评估实际平均递归次数1.8684084099807194；有效targets=50,753,524。
C相同边界NLL=3.537482451614582，但这不是完整语料结果或D/B比较。

16:11 UTC附近四个正式作业均终态，监控会话正常结束；controller仍保持原失败state，
没有手工改completed或伪造恢复。项目配额最新使用266,022,155,636 KiB/250TiB，
约2.25TiB空余；共享项目其他作业释放了部分空间，但仍不足以容纳下一轮全部
四组的既定checkpoint写入。此前清理授权请求尚无回复；没有删除数据、checkpoint，
没有擅自迁移到其他项目绕过配额，也没有启动新段。正式目标依然是四组5954更新，
不能把C/D520步或恢复工具测试通过当作约50B任务完成。

### 2026-09-19 获准清理、恢复状态与持久保留策略

用户明确回复“允许”，授权此前提出的旧权重清理及里程碑/最新两个恢复点保留策略。
先重查全部作业实际终态及远端文件，再执行精确清单，不用递归删除目录：

- 调优清理：10个已完成短跑，160个`.distcp`，2,565,126,228,958 bytes（约2.33TiB）。
  清单`formal-driver-v1/approved-tuning-prune-v1.json`，SHA256
  `c004e644f9514bbf1fb579bad41b924fac9f4c021881356412e8b65e8d14cc2c`；
  逐文件删除记录在同名前缀`.log`。先重验原50步产物和SLURM COMPLETED/0:0。
- 正式旧权重清理：A100、B100、C100/200/300/400、D100/200/300/400，
  共160个分片、2,076,506,850,206 bytes（约1.89TiB）。
  保留A/B120和200，C/D120、500和520；A不完整300也保留，不作为可恢复点。
  清单及逐文件操作日志保存在`formal-driver-v1/retention-<timestamp>.{json,log}`。

两轮合计约4.22TiB。只删除权重，保留全部JSON/日志、metadata、RNG、配置、数据和
签发证据；已删除权重不能直接恢复，只能重跑重建，不能把元数据留存称为模型备份。
`prune_weight_files.py`先验证整份清单的作用域、文件所有者、regular-file类型、
inode/size/mtime、非symlink、完整marker、原生分片命名及清单SHA，才逐文件unlink。
删除异常不自动盲重试，保留部分操作日志供核对。

正式清理前逐个检查保留点的native元数据引用范围及64份RNG文件CRC/SHA，确认
无缺失/截断；未读取全部tensor做内容checksum，不冒充新GPU恢复通过。
`reconcile_campaign.py --recover A:200 --recover B:200 --apply`已重新核对registry、
SLURM终态、远端checkpoint和历史输入，归档原失败state后应用A200/B200/C520/D520；
没有把checkpoint以后的失败更新计入有效tokens，也未补造完整作业结果。

`drive_campaign.py --prune-approved`现显式启用获准的保留策略，默认仍关闭，
仅此操作选项允许在同一锁下迁移binding，科学配置/receipt/source不变。
`maintain_checkpoints.py`在提交前及运行监控中每约5分钟检查：保留科学里程碑、
每组最新两个完整点及活跃作业resume源，忽略不完整点，不删除元数据；
跨组hash比较只使用失败作业已接纳的前缀，不读被丢弃或可能损坏的日志尾部。

新作业空间预算覆盖已有活跃/排队作业与候选段剩余的百步、科学里程碑和末步保存，
使用实测分片大小加10%，另留512GiB余量。可用字节读取Lustre quota-aware
`statvfs`，本次与`lfs quota -p 10089`差额逐字节核对一致。不足时等待且不提交，
不会取消已有作业。它不是配额预留，其他项目成员仍可能在检查后大量写入；
此限制不能被描述为保证永不发生配额错误。

CPU预训练回归140 passed、ruff通过。生产source仍58b909e3...、bundle仍08901c99...，
GBS2048、MBS2/2/4/2、各64GPU/EP16/segment4、5954步日程不变。
恢复审计需CPU PyTorch，controller仅通过已测Python3.12 CPU测试环境的site-packages
补充本地读取依赖；GPU任务仍使用原固定容器/venv和fused wrapper。
控制器已在宿主后台启动，日志`formal-driver-v1/driver-retention-host-v3.log`；
此时只能认定控制进程存活，实际提交和GPU恢复另按后续证据记录。

### 2026-09-19 提交前 SSH 故障与受限恢复

v3控制器在A200→600的`sync_project`同步`pyproject.toml`时遇到
`ssh_exchange_identification: Connection closed by remote host`，rsync返回12。
异常发生在`launch_remote_slurm`创建registry记录和调用远端提交入口之前；
本地/远端registry按唯一输出目录检索均为空，远端输出目录不存在。
没有新SLURM作业，不能把该失败称为训练恢复失败。

toolkit `cluster_run.remote.sync_with_retry`现在仅对包含明确SSH握手
closed/reset信息的rsync12沿用有限重试；普通rsync12、权限错误和sbatch
仍不自动重试。cluster_run回归30 passed。
独立`reconcile_presubmit.py`重审receipt/source/全部历史、下一个分段、
错误栈和两端registry后才允许归档旧intent并恢复ready；9项测试覆盖
已有job记录、输出目录、可疑提交日志和错误intent的拒绝路径。
本次dry-run和apply均通过，旧state和失败日志保留，随后启动v4控制器。
预训练CPU总回归149 passed。v4提交A时再次遇到一次握手中断，有限重试后
成功提交`7278726`（registry `20260918-224156-c160`）。2026-09-19 05:42 UTC
调度器确认PENDING、64GPU/16节点、SegmentSize4、4h、Dependency=null；
恢复参数指向A原作业checkpoint200，目标600。此时还不是GPU恢复通过。

B随后提交`7278761`，同样200→600、64GPU/segment4/4h、无依赖。
C的本地registry `20260918-224557-4d47`创建后，在调用远端入口的SSH认证前
握手中断，255/空stdout/精确identification错误，未返回远端job ID；控制器v4
因此停止而非盲重试。两端审计确认仅存在这条本地failed/null scheduler记录，
远端registry和目标目录为空。`reconcile_presubmit.py`增加这一精确失败分类，
要求本地日志与registry吻合且远端空白，保持A/B已提交记录不变。
恢复检查14项测试通过。toolkit仅对可证明未认证/exec的SSH握手断开有限重连，
一般连接中断/timeout/非空stdout/远端混合错误仍不重试，cluster_run共37项通过。
该修改不涉及训练source、科学配方或已签发验收证据。

v5控制器宿主PID352158，日志`formal-driver-v1/driver-retention-host-v5.log`。
C成功作业`7279034`（registry `20260918-225238-218e`），D`7279077`
（registry `20260918-225515-2789`），均520→920；A/B仍是前述作业，不重复提交。
2026-09-19 05:56 UTC四个作业均PENDING，均64GPU/16节点、SegmentSize4、
4h、无依赖；四条registry命令恢复点/MBS/output/stop_tokens逐项核对通过。
预训练CPU总回归154 passed、cluster_run37 passed、ruff通过。
尚未发生新GPU恢复验收；不能把成功排队称为50B完成。

2026-09-19 06:12 UTC补充：A `7278726`已实际从checkpoint200恢复，产生
201–203步完整训练记录。新manifest除topology外与checkpoint合同逐项一致，
64个rank输入日志首步均201，三步全局输入hash与原失败尝试相同步骤一致，
token计数连续、loss/grad_norm有限。203步累计1,702,887,424 tokens，
训练LM loss5.523457976989448、grad_norm0.4046764671802521，
LR0.0002996055740046054；这是训练batch loss，不是验证NLL。
启动首步41.13s，随后18.95/20.40s，不能据三步外推长期吞吐。
实际rank0–15/16–31/32–47/48–63分别位于block167/block104/block067/block118，
每组16rank/4节点且无跨domain。仅证明A当前GPU恢复及早期更新可用，
不等于600步分段或5954步全量完成；B/C/D此时仍排队。

2026-09-19 06:19 UTC：C `7279034`从checkpoint520恢复，521–524步序号与
token计数连续，64rank首步均521，新manifest合同与原checkpoint一致，
loss/grad_norm有限。524步训练LM loss3.5456531293457374，累计
4,395,630,592 tokens；该步5.80s，仅作启动观测而非长期吞吐。
四组EP rank0–15/16–31/32–47/48–63分别位于block101/037/122/088，
各4节点、16rank，未跨domain。B已RUNNING但恢复更新尚待确认，D仍排队。

2026-09-19 06:20 UTC：B `7278761`已从checkpoint200恢复，64rank首步201，
201–204序号/token连续、loss/grad_norm有限，输入hash及LR逐步与当前A相同；
新manifest合同与checkpoint一致。204步训练LM loss5.520484705222771、
grad_norm0.564863383769989，累计1,711,276,032 tokens。
四组EP分别位于block049/169/050/102，每组16rank/4节点、未跨domain。
A/B/C恢复已实证，D仍排队；新段保存、最终验证和全量完成尚待后续审计。

2026-09-19 06:29 UTC：D `7279077`已RUNNING并进入分布式初始化，实际恢复
更新尚待确认。C完成600步（5,033,164,800 tokens）保存，checkpoint600包含
16个distcp、183,219,663,613 bytes及step600完整标记，随后推进到604步。
这证明获准清理后的本次C保存流程已完成，不等于逐tensor内容checksum或新进程
恢复验证；分段终态仍由正式审计器检查。A/B也持续RUNNING，未因C保存而中断。

2026-09-19 06:33 UTC：D `7279077`实际加载dist_opt checkpoint520后完成
521–522步，64rank首步均521，新manifest合同与checkpoint一致；输入hash与
LR均匹配C的同一步，loss/depth_aux/grad_norm有限。522步训练LM loss
3.4191927940701135、depth_aux0.00040483914128230936，累计
4,378,853,376 tokens。四组EP位于block167/016/067/174，各16rank/4节点、
单domain。至此四组全部实际恢复并训练；不能据此替代各段保存、评估及50B终态审计。

自动保留策略首轮在线清理完成：`retention-1789799646187455522.json/.log`
精确删除C旧checkpoint500的16个distcp，183,219,663,613 bytes（约171GiB）；
保留C科学里程碑120、当前resume源520及最新600。日志/metadata/RNG未删，
删除权重不可直接恢复。操作后可用10,537,510,993,920 bytes，剩余活跃段
写入预算5,050,927,195,610 bytes，空间准入通过；其他项目写入仍可能改变余量。

2026-09-19 06:47 UTC：A `7278726`在新输出目录完成checkpoint300，16个
distcp总427,529,457,929 bytes且step300完整标记已生成，随后成功推进301步。
这是此前A因配额失败的保存边界，本次没有重现配额错误；原失败目录里的不完整300
仍保留，不与新完整300混淆。此检查证明保存流程完成并继续训练，不是逐tensor
checksum或该新checkpoint的外部进程恢复证明。C此前700也完成16分片及完整标记
并继续705+；四组仍在进行本轮分段，50B目标未完成。

2026-09-19 06:52 UTC：B `7278761`也完成checkpoint300（16个distcp，
183,219,663,613 bytes及step300完整标记），随后推进302步；已跨过之前
日志写入因配额失败的299步边界。C800亦完成同样分片大小/完整标记并推进804+。
这些是保存和持续更新的现场证据，完整分段审计及新checkpoint恢复验证仍另计。

2026-09-19 07:07 UTC 复核：C `7279034` 的520→920段已以`COMPLETED 0:0`
退出（SLURM用时50分26秒），控制器完成段末审计并将C已完成游标更新为920。
累计7,717,519,360输入tokens，固定验证集50,753,524个targets的LM NLL为
3.091605370215014、PPL为22.012387657130756，策略`causal-full-depth`；
`complete.jsonl`标记`full_epoch=false`。该结果不是10B或50B最终结果，
也不能与其他组不同token进度的即时训练loss作配对比较。训练版本与上述source58合同不变。
现场A/B/D分别推进至365/353/614步，控制器仍存活；C下一段应由控制器独立接续到1193，
此处尚未将下一段记为已提交。

在线清理凭据`retention-1789800811148827412.json/.log`记录删除C旧checkpoint600
的16个distcp，共183,219,663,613 bytes；当时保留科学里程碑120、活跃resume源520
及最新700/800。权重不可直接恢复，日志、metadata、RNG保留。

2026-09-19 07:11 UTC：控制器自动提交C的920→1193段，SLURM `7279982`，
本地registry `20260919-001045-f422`，远端registry `20260919-001048-44a1`。
现场`scontrol`确认为PENDING、16节点/64 GPU、`SegmentSize=4`、4小时时限、
`Dependency=(null)`；并非等待A/B/D完成才启动。目标累计10,007,609,344输入tokens。
提交前在线保留策略凭据`retention-1789801709470207614.json/.log`精确删除
C旧520/700/800和D旧500的共64个distcp，732,878,759,392 bytes；C520所属
作业已经完成，后续恢复源为920。元数据及日志仍保留，所删权重不可直接恢复。
这里只证明提交和资源配置，尚不证明新作业已经加载checkpoint或完成训练。

随后C `7279982`实际完成921–923步：64个`input-rank-*.jsonl`首步均为921，
新manifest除topology外与920完整checkpoint的contract逐字段一致。
921步累计7,725,907,968 tokens、LR 0.00028601884826470845、训练LM loss
3.0938177770003676、grad norm 0.15387123823165894；923步仍有限并继续更新。
这是该续训进程恢复并执行更新的证据，不等价于全部参数/optimizer逐tensor校验，
也不代表1193或5954步完成。

续训在线一致性抽查（A/B/C/D分别408/401/980/655步）：仅使用已接受的正式训练
前缀，排除A/B旧失败尝试200步之后的尾部，各组步骤1到当前步无缺口。
逐步比较`input_sha256`、`lr`、`tokens`，A/B、A/C、A/D、B/C、B/D、C/D
分别有401/408/408/401/401/655个重叠步，均无不一致。该检查覆盖当时已有日志，
不预先保证未来步骤，也不代表梯度或模型输出相等。A400和B400均已生成完整标记
及16个distcp（分别427,529,457,929和183,219,663,613 bytes），保存后继续更新。

在线清理凭据`retention-1789802916012257325.json/.log`删除C旧900的16个distcp，
183,219,663,613 bytes；C已完成1000保存后继续更新，活跃恢复源为920。
日志与metadata/RNG未删，所删权重不可直接恢复。清理后控制器记录可用
9,597,125,947,392 bytes、剩余活跃段写入预算3,102,654,122,860 bytes，准入通过。

### 2026-09-19 只读盘点超时导致控制器退出的修复

v5 PID352158在`remote_inventory`的SSH只读目录/marker/分片stat盘点超过60秒后退出；
错误类型为`TimeoutExpired`，不是GPU训练失败，也不是删除或sbatch结果不确定。
现场四个作业`7278726/7278761/7279982/7279077`仍RUNNING，C持续推进至1191步。
不能仅从超时判断具体延迟位于SSH还是Lustre；已证明的控制器缺陷是把单次只读观测
超时直接作为终止条件，未做有界重试。

`plan_checkpoint_retention.remote_inventory`现在仅对此只读命令的TimeoutExpired
做最多3次尝试（60/120/180秒，间隔5/10秒）。远端校验失败、非法JSON仍立即失败；
没有给删除、提交或整个maintenance套重试。所有盘点失败仍停止，不以缺失盘点通过准入。
`drive_campaign --recover-inventory-timeout`是显式恢复入口：仅接受schema2、无intent、
仍有active jobs且error逐字匹配当前显式作业目录盘点命令的失败状态；保留原错误记录，
重审receipt/source/bundle/历史证据和阶段游标后，归档旧state，恢复watching。
其他失败（包括sbatch和prune超时）不能用该入口恢复，活跃job ID/游标不变。

相关33项测试、全部157项预训练CPU回归及ruff通过。远端重新只读盘点返回35条记录，
保留策略校验通过、无候选删除；生产source58b909e3...及bundle08901c99...不变。
修复后的v6宿主PID422595，日志`formal-driver-v1/driver-retention-host-v6.log`；
已通过启动审计，state恢复watching，接管原四个job ID，没有重启GPU作业或重复提交。
这些证据仅恢复自动推进能力，不代表四组10B或50B完成。

### C 的10B阶段完成（其余组尚未到同一token边界）

`7279982`最终`COMPLETED 0:0`，SLURM用时34分29秒。v6控制器已完成原有
段末审计并更新C完成游标1193、移出active；不是仅凭日志或队列状态宣称通过。
累计10,007,609,344输入tokens，固定验证集50,753,524个有效targets：
token-weighted LM NLL=2.955054697466526、PPL=19.202773034143373，
策略`causal-full-depth`。920→1193段实际记录34.69468934612969 GPU-hours，
不含C之前各段，不可当作累计10B总成本。最终checkpoint1193和完成产物已归档审计。
A/B/D仍训练，按原方案阶段门槛，C等待四组均完成10B后再进入下一阶段；
不能拿C10B与其余组不同进度即时loss作效果结论，50B目标未完成。

控制器本轮保留策略凭据`retention-1789804247358721949.json/.log`删除C旧1000
的16个distcp，183,219,663,613 bytes；日志/metadata/RNG保留，所删权重不可直接恢复。
空间准入记录可用9,225,975,267,328 bytes、剩余活跃段写入预算2,498,029,117,503 bytes。

后续在线清理凭据`retention-1789804846861601531.json/.log`删除新续训目录的
A300、B300及已完成C作业的旧resume源920，共48个distcp、793,968,785,155 bytes。
A/B已完整保存500并继续，C已通过1193段末审计、不再以920作为活跃恢复源；
原A失败目录中的不完整300不在此次清单内。日志/metadata/RNG保留，所删权重不可直接恢复。
本轮可用10,056,028,266,496 bytes、剩余活跃段写入预算1,826,205,083,807 bytes，准入通过。

在线清理凭据`retention-1789806108859097795.json/.log`删除D旧600的16个distcp，
183,219,768,553 bytes；D800已完整保存并继续更新，活跃恢复源520受保护。
日志/metadata/RNG保留，所删权重不可直接恢复。控制器记录可用9,983,572,553,728 bytes，
剩余活跃段写入预算1,624,663,338,399 bytes，准入通过。

B `7278761`的200→600段已`COMPLETED 0:0`（2h12m28s），v6控制器完成原有
段末审计，B完成游标更新为600。累计5,033,164,800输入tokens，固定验证集
50,753,524个targets的NLL=3.4886773368724957、PPL=32.742611607948035。
600完整checkpoint含16个distcp、183,219,663,613 bytes。A同边界checkpoint
也已完整保存（16分片、427,529,457,929 bytes），但此时A的最终评估/终态尚待确认。
这不是四组10B或50B完成，也不拿B5B与C10B作等token比较。

A `7278726`最终`COMPLETED 0:0`（2h21m29s），v6控制器完成段末审计并更新
A游标600；同5,033,164,800输入tokens和50,753,524有效targets，验证NLL
3.4245000198575455、PPL30.707287978539153。A/B600阶段均有完整正式审计。
B下一段600→1000已自动提交为`7280773`，registry `20260919-013301-0899`，
目标累计8,388,608,000 tokens，从原B600恢复。提交前rsync发生一次临时故障，
已有受限重试后成功；不是训练失败，也没有重复sbatch。A后续提交仍待核实。

### checkpoint只读校验的SSH故障恢复

v6 PID422595在A下一段提交前、核验D800保留恢复点的`remote_files`时因SSH
返回255退出；该异常发生在只读元数据/RNG哈希与CRC查询，不是GPU失败、删除或
提交失败。旧异常日志没有保留stderr，故不能断言这次255的具体SSH子原因。
B `7280773`已在队列，D `7279077`仍训练，A尚无新intent/提交，不能重复提交B。

新增`readonly_probe.run_readonly`仅供明确的只读探针使用：SSH255或TimeoutExpired
最多3次尝试（60/120/180秒，5/10秒间隔），重试输出类型和次数；远端校验失败
（如CRC损坏、非法文件）不重试。`inspect_recovery.remote_files`接入该工具。
显式`--recover-checkpoint-read NATIVE_PATH`只匹配当前campaign目录内、原错误中
完全一致的只读命令，拒绝intent/提交/删除异常，并必须重新读取成功后才进入原有
receipt/source/bundle/历史审计和state归档流程。没有放宽存储完整性检查。

全部161项预训练CPU回归、ruff通过；实际D800只读元数据/RNG检查重跑成功，
production source58b909e3...、bundle08901c99...未变。v7控制器宿主PID454669，
日志`formal-driver-v1/driver-retention-host-v7.log`，接管现有B/D，不重启GPU作业。
仍需逐项验证后续提交及全部5954步，不能把本次控制器恢复当作训练完成。

v7随后在第二次目录盘点遇到SSH255退出，暴露盘点只覆盖TimeoutExpired而遗漏
SSH传输错误的剩余缺陷；D持续训练、B仍排队，A尚未提交。现将目录盘点、
checkpoint小文件校验、quota-aware可用空间读取及提交前receipt/wrapper哈希查询
统一接入`run_readonly`。删除、sbatch、混合读写maintenance不套此重试；
远端校验错误/非法JSON不重试。显式`--recover-inventory-read`（旧timeout flag别名）
仅接受完全一致盘点命令的255/timeout，归档并重审历史，不能恢复其他副作用失败。
全部163项预训练CPU回归及ruff通过；远端重新盘点40条记录并通过保留策略校验，
候选删除0。v8宿主控制器PID458964，日志`formal-driver-v1/driver-retention-host-v8.log`，
继续原配置和已提交B/D，训练生产源码/配置没有修改。

v8已自动提交A600→1000，SLURM `7280872`、registry `20260919-014951-84f3`，
输出`formal-source58-v1-a-000600-001000`。实查`scontrol`为64 GPU、16节点、
`SegmentSize=4`、4小时、无dependency；作业名`mor-a-mbs2`。A/B下一段均
PENDING，D `7279077`继续RUNNING，C1193等待原定10B阶段屏障；尚非50B完成。

补记已执行的精确清理凭据：`retention-1789806627421521277`删除旧B200、
新A400/B400共48个distcp（793,968,785,155 bytes）；
`retention-1789807348537198973`删除旧A200共16个distcp（427,529,457,929 bytes）。
A/B600均已完整保存、原段已通过审计，后续恢复源为600；里程碑及最新两个完整
checkpoint仍保留。上述仅删除权重分片，不删除日志、metadata或RNG；所删权重
不可直接恢复。v8最新空间准入记录可用10,584,941,584,384 bytes、预算
1,759,005,824,601 bytes，准入通过；该余量不是对共享存储空间的预留。

D `7279077`已保存900步并继续训练至908步。900目录含16个distcp，合计
183,219,768,553 bytes，metadata、64份rank RNG文件及pretraining-state均存在；
state步数900、数据游标7,549,747,200、scheduler为`pure-token-function-v1`，
source/data/environment合同仍为本次正式版本。这里只验证文件布局、字节数和合同，
不冒充该点的逐tensor校验或新进程恢复验证。
将D实时`loss.jsonl`与已审计C `7279034`的证据逐步比较，521–908共388步
连续无缺口，`input_sha256`、`tokens`、`lr`全部完全一致；模型loss不要求一致。

v8在后续轮询中两次只读查询触发SSH255受限重试，随后正常完成maintenance，
进程仍存活；本次观察表明重试覆盖了实际短暂故障，并非重启GPU作业。
`retention-1789808493757871432.log`确认清理D700的16个distcp，
183,219,768,553 bytes；D800/900和活跃恢复源520保留，元数据及日志未删，
已删权重不可直接恢复。最新可用10,582,559,580,160 bytes，写入预算
3,438,593,694,080 bytes，准入通过。

### 段末证据下载的SSH故障修复

D `7279077`的520→920段实际`COMPLETED 0:0`（2h39m29s），累计
7,717,519,360 tokens；固定验证50,753,524个targets的NLL
3.055644183143615、PPL21.234860260959696，`strict-causal-threshold`策略下
平均递归次数1.920227131119206。本段记录167.50778301646312 GPU-hours，
不是累计训练成本，也不是10B/全量完成。

v8随后因段末证据rsync下载遭遇`ssh_exchange_identification: Connection closed
by remote host`、退出255而停止；已确认宿主PID458964消失。此前只读SSH探针重试
未覆盖这条“远端只读、本地生成证据镜像”的传输路径，导致训练正常结束仍无法自动接续。
A/B原有排队任务不受影响，D完成游标暂时仍520，不能手工填920绕过审计。

`drive_campaign.fetch_evidence`现仅对此固定方向、固定包含规则的下载，针对255
或timeout最多尝试3次（180/240/300秒，间隔5/10秒）；不删除远端、不上传，
普通rsync错误和完整性/科学审计错误不重试。`--recover-evidence-transfer JOB_ID`
只接受当前active job、无intent且error逐字匹配原下载命令的失败状态，必须重新
下载成功才恢复；原state归档、receipt/source/bundle/历史审计仍保留，主循环重查
SLURM终态并执行原段末审计后才能推进completed。未修改GPU训练源码、模型或配置。
全部170项预训练CPU测试及ruff通过，覆盖有界重试、非传输错误、错误job/目录、
提交/删除异常、失败重读的拒绝，以及active/已完成游标不被恢复函数改写。

v9宿主PID473355，日志`formal-driver-v1/driver-retention-host-v9.log`；
实际重拉证据并通过启动科学绑定检查及D原有段末审计，输出`audited: 7279077`，
完成游标现为A600/B600/C1193/D920，state为watching、无intent/error。
A/B仍保留原scheduler ID，不重复提交；D下一段920→1193尚待控制器提交确认。

v9在D段末审计后执行`retention-1789809172122785302`，删除已结束作业的旧
恢复源D520和旧D800，共32个distcp、366,439,537,106 bytes；科学里程碑及
最新D900/920保留。日志/metadata/RNG未删，所删权重不可直接恢复。
可用10,738,577,920,000 bytes、活跃段写入预算3,237,051,948,672 bytes，准入通过。

D920→1193已自动提交为SLURM `7281040`，本地registry `20260919-021427-326d`，
远端registry `20260919-021432-6579`；恢复源为原D完整checkpoint920，目标
10,007,609,344 tokens，仍使用bundle08901c99...。实际`scontrol`确认
64GPU/16节点、SegmentSize4、4h、无dependency，提交命令MBS2。当前A/B/D
均PENDING，尚无这次D的GPU恢复证据；C已完成1193等待阶段屏障。

### 2026-09-19 清理命令认证前断连的受限恢复

A7280872/B7280773已从600步实际恢复并保存700步，训练继续；D7281040仍排队。
控制器v9 PID473355在保留策略删除调用返回SSH255后退出。失败凭据
`retention-1789828015423133026.json`的SHA256为
`549cb7932c4273f82f8ba672c2519ef32011bcad53be95b77ca8e9609c0eb5f8`，
目标为旧B500的16个distcp；配套日志仅含
`ssh_exchange_identification: Connection closed by remote host`，没有远端删除输出。
只读重新盘点及原删除程序的非apply校验确认全部目标的size/inode/mtime仍匹配，
仍是当前保留策略候选，state文件未改；不是以SSH255笼统推断“没有副作用”。

`maintain_checkpoints.run_prune`现在分别捕获stdout/stderr，仅对空stdout、255和
严格匹配的OpenSSH认证前握手断连最多尝试3次，间隔5/10秒。超时、一般连接断开、
远端错误、任何删除输出或混合错误均不重试；每次返回的输出保留在清理日志中。
新增`--recover-prune-preauth MANIFEST_NAME`只接受原失败命令、manifest哈希和
认证前日志完全匹配的已授权campaign；重新盘点、核对候选身份并执行只读验证后，
才进入原receipt/source/bundle/历史审计与state归档。恢复本身不删除文件、不提交任务，
后续maintenance必须重新生成保留计划并验证保留点，不复用旧清单直接删除。
保留原active job ID和completed游标，不改训练源码、模型、数据或学习率日程。

全部187项预训练CPU测试通过，修改文件ruff检查通过；新增17项测试覆盖受限重试、
超时/部分输出/远端错误拒绝、目标变更/缺失/受保护时拒绝、只读恢复及状态不变。
这次修复仅恢复控制器的持续运行能力，不代表1000步段末或完整50B验收完成。

部署状态：v10持久化启动命令在执行前被安全审核拒绝，理由为恢复控制器会带来
远端checkpoint删除副作用，需明确授权。未绕过审核，未启动v10；v9已退出，
现有A/B GPU作业继续运行、D仍排队。187项测试及只读核验通过不等于控制器已恢复。

### 可重复的下一步入口

2026-09-19 用户要求不因清理任务停止推进。新增显式
`--retain-all-checkpoints`：maintenance仅盘点与计算空间预算，不执行权重删除，
即使旧状态含prune授权也不删除；提交前空间不足仍等待，不跳过准入或训练审计。
该开关仅改变控制器保留策略，不改变生产源码、数据、模型、保存频率和科学合同。
恢复仍须核实原失败清单无副作用并重审历史，不能手工将failed改为watching。
新增测试覆盖存在清理候选时零删除、足够空间准入及不足空间拒绝。
此模式会积累checkpoint，不能据当前余量保证全50B无需后续空间处理。

随后用户明确授权仅在空间不足时删除多余checkpoint。新增显式
`--prune-on-pressure --prune-approved`，与retain-all互斥；先运行不删除的盘点和
预算，空间足够立即返回零删除。预算不足才执行原精确清单验证和保留策略清理，
之后重新读取可用空间，仍不足则不准入新任务。活跃恢复源、里程碑、每组最新两个
完整点和元数据保留规则未变。此检查不预留共享空间，也不保证其他用户不会耗尽磁盘。
测试覆盖足够空间零删除、缺口触发、清理后仍不足，以及缺少授权/模式冲突的拒绝。

从主 toolkit checkout 运行（路径为本次已有工作树，不改共享 vendor）：

```bash
PYTHONPATH=.claude/worktrees/mor-mlite-validation/vendor/mor_mlite/src:src \
  .venv/bin/python -m mor_mlite.pretraining.launch \
  --arm B --mbs 1 --world-size 32 --smoke \
  --native-root runtime/mor-pretraining-native-5c8315f \
  --python '${PROJECT_ROOT}/runtime/mor-pretraining/gb200-env-nvrx060/bin/python' \
  --output '${PROJECT_ROOT}/runtime/mor-pretraining/<new-smoke-output>' \
  --container-sqsh /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/denliu/agentic-mcore-dev/runtime/containers/gb200-torch2603_231f7a2ae3a0.sqsh
```

上面默认仅 render；确认产物后追加 `--submit`。输出目录必须是新目录。
原 smoke 全部成功后，在另一次进程启动中追加
`--resume '${PROJECT_ROOT}/runtime/mor-pretraining/<completed-smoke-output>'`，
并使用另一个新 `--output`，才是外部进程连续性测试。
非 smoke 的 tuning 需提供 `--data <data-50b/prepared>`、
`--hf-config <data-50b/tokenizer>`、`--environment <candidate-environment.json>`。
正式训练还需真实 `--acceptance`；目前不要创建或填充未完成验收项。
