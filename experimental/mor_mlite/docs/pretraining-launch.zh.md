# Qwen3 四组预训练：MBS 与启动验收

2026-09-19 22:35 UTC，修复独立组空间准入队首阻塞后的v13自动提交
D1593→1993为7297954（registry 20260919-223553-1b98），scontrol核验
16节点/64GPU/SegmentSize4/MBS2/4h/Dependency=null，PENDING。
A1993→2393仍被原空间预算拒绝，但不再阻挡D；B7297442/C7296632原ID保留。
宿主控制器PID1117475、watching、无intent/error；197项预训练CPU回归及ruff通过，
source58/bundle08901不变。详见pretraining-design.zh.md的调度修复记录。

2026-09-19 D7289784 的1193→1593段 COMPLETED、ExitCode=0:0，elapsed=02:40:44
（batch/extern同，step0=02:40:25，均0:0），控制器已audited、completed.D=1593。
累计输入13,363,052,544 tokens，固定验证50,753,524 targets：NLL=2.78984023065424、
PPL=16.278418801691966，strict-causal-threshold，实际mean_recursions=
1.982547799440014；估算评估forward FLOPs=2.7369261863303168e17。
验证输入50,765,919、排除chunk边界targets12,395、无丢弃验证尾部；本段wall_seconds=
9493.331263205968、GPU-hours=168.77033358972105（非累计成本）。full_epoch=false。
1593保存点16个非空distcp共183,219,768,553 bytes、64份非空RNG、非空.metadata，
完成标记/合同及cursor=13,363,052,544通过结构检查，非全tensor校验。
另逐步独立比较ABCD1194–1593全部400更新的全局input_sha256、tokens和LR，完全一致。
压力清单retention-1789881834032713856.json/.log删除D1400的16个distcp，
183,219,768,553 bytes；不可恢复，未删数据集/日志。远端确认分片已不存在，
D1500/1593仍各16个非空distcp、64份非空RNG、非空.metadata及完成标记。

2026-09-19 A1993→2393 暂缓提交：空间准入需要5,588,408,281,369 bytes，
压力清理后仅4,106,208,104,448 bytes（随后4,127,470,166,016），admit=false。
预算来自现有B/C/D剩余保存，加A的2000/2100/2200/2300/2385（20B）/2393六次
保存，按实测分片大小加10%及512GiB安全余量；不是训练失败，未绕过检查。
控制器仍存活并重试准入，D继续训练、B/C保持排队。
清单 formal-driver-v1/retention-1789880683276009917.json 及同名.log 确认
删除A1593/1800的32个distcp，共855,058,915,858 bytes；分片不可恢复，未删
数据集或日志。独立远端确认已删除，并核验9个保护点仍各有16个非空distcp、
64份非空RNG、非空.metadata及完成标记：A1900/1993，B1500/1593，C1500/1593，
D1193/1400/1500。此为文件结构检查，不是全tensor校验。

2026-09-19 A7292218 的1593→1993段已由sacct确认COMPLETED、ExitCode=0:0，
elapsed=02:18:18（batch/extern同，step0=02:17:58，均0:0）；控制器已audited并
更新completed.A=1993。400次更新连续且loss/grad_norm/LR有限，累计输入
16,718,495,744 tokens；固定验证集50,753,524 targets 的NLL=2.660385711636013、
PPL=14.30180440732575，policy=causal-full-depth。验证输入50,765,919，
排除chunk边界targets12,395，无丢弃验证尾部；估算评估forward FLOPs
3.906091846926336e17。本段wall_seconds=8134.963625979144、GPU-hours=
144.6215755905956（非累计成本）。complete.full_epoch=false，目标仍为5954步。
1993保存点结构/合同核验：16个非空distcp共427,529,457,929 bytes、64份非空RNG、
非空.metadata、完成标记及cursor=16,718,495,744正确；非全tensor校验。
下一段提交待确认，D仍运行、B/C排队。最新空间3,272,718,184,448 bytes高于
预算2,766,713,859,038 bytes，未新增清理。

2026-09-19 21:57 UTC，B1593→1993 段自动提交为7297442
（registry 20260919-215724-336c）。独立scontrol确认16节点、64GPU、
SegmentSize=4、MBS=2、TimeLimit=04:00:00、Dependency=(null)，当前PENDING。
控制器已回到watching，无error；A7292218/D7289784仍RUNNING，C7296632仍排队。
新段尚未开始，不将提交成功当作跨进程恢复成功。

2026-09-19 为 B 后续分段进行空间维护：清单/执行日志
formal-driver-v1/retention-1789880139080959116.json 及同名.log 确认删除
B1300/1400、D1200/1300、A1700 的80个distcp，共1,160,408,322,261 bytes，
applied=true；这些分片不可恢复，未删除数据集或日志。独立远端确认五处
distcp已不存在，同时11个保护点仍各有16个非空distcp、64份非空RNG、非空
.metadata及完成标记：A1593/1800/1900，B1193/1500/1593，C1500/1593，
D1193/1400/1500。这是结构核验，不是全tensor校验。控制器正在submitting，
B下一段作业ID尚待确认。

2026-09-19 B7289637 的1193→1593段已由 sacct 确认 COMPLETED、ExitCode=0:0，
elapsed=02:09:14（batch/extern同，step0=02:08:54，均0:0）。400次更新结束，
累计输入13,363,052,544 tokens；固定验证集50,753,524 targets 的
NLL=2.8252696952812433、PPL=16.86549291663922，policy=causal-full-depth，
验证输入50,765,919、排除chunk边界targets12,395、无丢弃验证尾部。
估算评估forward FLOPs=3.906091846926336e17；本段wall_seconds=7576.174615080934，
GPU-hours=134.68754873530733（非累计成本）。complete记录full_epoch=false，
目标仍为5954步。1593保存点16个非空distcp共183,219,663,613 bytes、64份非空
RNG、非空.metadata、完成标记及contract/cursor=13,363,052,544通过文件结构检查，
非全tensor校验。控制器已 audited 7289637、completed.B=1593，无error；下一段
提交仍待确认。A/D仍RUNNING、C7296632排队。

2026-09-19 D7289784 的 1500 checkpoint 完成文件结构/合同核验，并继续到1501：
16个非空 distcp 共183,219,768,553 bytes、64份非空 RNG、一份非空 .metadata；
完成标记 step=1500、cursor=12,582,912,000，contract 与 manifest 去掉 topology
后一致。这不是全 tensor 校验。A/B/D 仍 RUNNING，C7296632 仍 PENDING。

2026-09-19 A7292218 的 1900 checkpoint 完成文件结构/合同核验，并继续到1901：
16个非空 distcp 共427,529,457,929 bytes、64份非空 RNG、一份非空 .metadata；
完成标记 step=1900、cursor=15,938,355,200，contract 与 manifest 去掉 topology
后一致。这不是全 tensor 校验。A/B/D 仍 RUNNING，C7296632 仍 PENDING。

2026-09-19 B7289637 的 1500 checkpoint 完成文件结构/合同核验，并继续到1502：
16个非空 distcp 共183,219,663,613 bytes、64份非空 RNG、一份非空 .metadata；
完成标记 step=1500、cursor=12,582,912,000，contract 与 manifest 去掉 topology
后一致。这不是全 tensor 校验。A/B/D 仍 RUNNING，C7296632 仍 PENDING。

2026-09-19 D7289784 的 1400 checkpoint 完成文件结构/合同核验，并继续到1405：
16个非空 distcp 共183,219,768,553 bytes、64份非空 RNG、一份非空 .metadata；
完成标记 step=1400、cursor=11,744,051,200，contract 与 manifest 去掉 topology
后一致。这不是全 tensor 校验。A/B/D 仍 RUNNING，C7296632 仍 PENDING。

2026-09-19 控制器按空间压力策略清理 A1600、B1200 的32个 distcp，
共610,749,121,542 bytes；清单/执行日志为 formal-driver-v1/
retention-1789876779597375163.json 及同名.log，applied=true。
独立远端确认两处 distcp 已删除，分片不可恢复，未删数据集或日志。
清理后空间3,986,860,793,856 bytes高于预算3,505,737,267,374 bytes。
另核验11个保护点的16个非空distcp、64份非空RNG、非空.metadata及完成标记：
A1593/1700/1800，B1193/1300/1400，C1500/1593，D1193/1200/1300。
上述为文件结构检查，不是全tensor校验；当前恢复源仍保留。

2026-09-19 A7292218 的 1800 checkpoint 完成文件结构/合同核验，并继续到1806：
16个非空 distcp 共427,529,457,929 bytes、64份非空 RNG、一份非空 .metadata；
完成标记 step=1800、cursor=15,099,494,400，contract 与 manifest 去掉 topology
后一致。这不是全 tensor 校验。A/B/D 三作业继续 RUNNING（约1:16运行时间），
C7296632 仍 PENDING；只读查询重试未被当作作业失败，也未触发人工重启。

2026-09-19 B7289637 的 1400 checkpoint 完成文件结构/合同核验，并继续到1406：
16个非空 distcp 共183,219,663,613 bytes、64份非空 RNG、一份非空 .metadata；
pretraining-state.json 标记 step=1400、cursor=11,744,051,200，contract 与本段
manifest 去掉 topology 后严格一致。此前仅出现 step_1400 目录时未将其认定为
完成保存点。这是保存结构/合同检查，不是全 tensor 校验。C7296632 仍在排队。

2026-09-19 C7289665 完成段补充数据一致性核验：1194–1593 全部 400 次更新
的全局 input_sha256、tokens、LR 均与 A7289602 同 step 严格一致，覆盖
C MBS=4 与 A MBS=2 的整个已完成段。同期 B1194–1372、D1194–1336 的
全局输入摘要亦与 A 一致；A 新段至1758，三组更新连续且 LM loss、grad_norm、
LR 有限。此结论限于已核验区间，不代表全程5954步已通过。

2026-09-19 20:37 UTC，C 的 1593→1993 段已自动提交为 7296632
（registry 20260919-203730-4532）。独立 scontrol 核验为 16 节点、64 GPU、
SegmentSize=4、MBS=4、TimeLimit=04:00:00、Dependency=(null)，目前 PENDING，
尚不能据此认定恢复启动成功。控制器回到 watching，无 submission_intent 或 error；
A7292218、B7289637、D7289784 仍 RUNNING。最近空间检查 free=3,948,237,004,800
bytes、required=3,169,853,151,199 bytes，admit=true、removed_bytes=0。

2026-09-19 C7289665的1193→1593段已由sacct确认COMPLETED、ExitCode=0:0，
elapsed=00:48:40（extern00:48:45、step0 00:48:15，均0:0）。累计输入
13,363,052,544 tokens，固定验证集50,753,524 targets：NLL=2.8241003256239394、
PPL=16.845782447625275，policy=causal-full-depth；验证输入50,765,919、
排除chunk边界targets12,395、无丢弃验证尾部。评估估算forward FLOPs
1.8118317036437504e17，本段wall_seconds=2752.7401101458818、
GPU-hours=48.937601973513765（非累计成本）。complete记录full_epoch=false。
控制器已audited 7289665并将completed.C更新为1593；下一段提交仍待核验。
同期A/B/D继续至1722/1335/1306。最终5954步目标尚未完成。

2026-09-19 C7289665已到1593，本段1194–1593共400次更新连续且指标有限；
段末checkpoint文件结构/合同核验通过：16个非空distcp共183,219,663,613 bytes，
64份非空RNG、非空.metadata，step=1593、cursor=13,363,052,544，contract一致。
当时尚无evaluation/cost/complete记录，Slurm仍RUNNING，不能宣布本段验收完成。
D7289784的1300 checkpoint亦通过相同检查（183,219,768,553 bytes，
cursor=10,905,190,400），已继续到1302。A/B至1717/1330，四组仍RUNNING。
这些是文件结构/合同核验，非全tensor校验；source58/bundle08901配置未改。

2026-09-19 A7292218的1700保存通过文件结构/合同核验并继续至1702：
16个非空distcp共427,529,457,929 bytes、64份非空RNG、非空.metadata；
完成标记step=1700、cursor=14,260,633,600，contract与manifest去掉topology
后一致。非全tensor校验。同期B/C/D至1314/1556/1291，四组仍RUNNING，
本段更新连续、loss/grad_norm/LR有限。控制器空间检查4,314,692,001,792 bytes
高于剩余预算3,572,936,526,582 bytes，未新增清理；source58/bundle08901不变。

2026-09-19 B7289637的1300及C7289665的1500保存通过文件结构/合同核验：
各16个非空distcp共183,219,663,613 bytes、64份非空RNG、非空.metadata；
完成标记step分别1300/1500，cursor分别10,905,190,400/12,582,912,000，
contract与各自manifest去掉topology后一致。非全tensor校验。
保存后B/C继续至1303/1522，同期A/D至1696/1282；四组仍RUNNING，
本段更新连续、loss/grad_norm/LR有限。配置source58/bundle08901未改。

2026-09-19 C7289665的1400保存完成并继续至1418：16个非空distcp共
183,219,663,613 bytes、64份非空RNG、非空.metadata；完成标记step=1400、
cursor=11,744,051,200，contract与manifest去掉topology后一致。
此为文件结构/合同核验，非全tensor校验。同期A/B/D运行至1662/1270/1255，
四组本段更新连续且loss/grad_norm/LR有限，控制器701996存活。

2026-09-19 20:09 UTC v12再次按压力清理：
`runtime/mor-pretraining/formal-driver-v1/retention-1789873773767541280.json`
及同名.log确认删除A1500、B/C/D1100共64个distcp，977,188,553,708 bytes。
权重/optimizer分片不可恢复，未删数据或日志。独立远端只读核验四处distcp
已不存在；A1593/1600、B/C/D1193/1200及C1300共9个保护点各仍有16个非空
distcp、64份非空RNG、非空.metadata及完成标记。清理后控制器可用空间
5,514,935,136,256 bytes，高于剩余预算4,647,843,820,227 bytes。
source58/bundle08901及训练配置未改，四作业仍RUNNING至A1655/B1262/C1400/D1249；
C1400当时尚在保存，不能将未出现完成标记的保存点视为恢复源。

2026-09-19 C7289665的1300保存完成并继续至1313：16个非空distcp共
183,219,663,613 bytes、64份非空RNG、非空.metadata；完成标记step=1300、
cursor=10,905,190,400，contract与manifest去掉topology后一致。
此为文件结构/合同核验，非全tensor校验。同期A/B/D仍RUNNING至1627/1233/1226，
四组本段更新连续、loss/grad_norm/LR有限；控制器701996存活，空间
5,074,472,939,520 bytes高于剩余预算4,647,843,820,227 bytes，未新增清理。

2026-09-19 当前四个作业运行约11分钟后，A1600、B/C/D1200保存通过
独立文件结构/合同检查：各16个非空distcp、64份非空RNG、非空.metadata，
完成标记step及contract与本段manifest（去掉topology）一致。
A分片427,529,457,929 bytes、cursor=13,421,772,800；B/C各
183,219,663,613 bytes、D183,219,768,553 bytes，后三组cursor=10,066,329,600。
保存后A/B/C/D已分别继续到1604/1209/1247/1207；本段更新连续，
loss/grad_norm/LR有限。配置仍为source58/bundle08901、64GPU、EP16、segment4、
MBS2/2/4/2。这是保存结构及继续训练验证，不是全tensor校验或再次跨进程恢复。

2026-09-19 恢复首步后续核验：A首个更新1594，LM loss=2.791171963501256；
四组各64份input-rank日志首步分别为A1594、B/C/D1194，已观察的loss更新
连续且loss/grad_norm/LR有限。B1194–1200、C1194–1212、D1194–1199的
全局input_sha256逐步与A上一段同step一致，B/D的64个rank首步hash亦一致。
C的单rank hash不要求与A一致：MBS4与MBS2改变microstep/rank的样本分配；
train.py按全局sample编号收集每行摘要，data.py::global_input_digest检查
完整连续GBS编号、拒绝重复或缺失，再按全局顺序计算摘要，因此应使用此
全局摘要比较跨MBS数据一致性。上述检查不是最终5954步验收。

2026-09-19 四组7292218/7289637/7289665/7289784均已启动。
独立检查各manifest全部64个rank记录：每个EP group恰有16 rank、4主机、
一个NVLink domain，ep_members一致。按rank 0/16/32/48起始的组，A为
block006/006/112/112，B为block167/167/043/043，C为block033/033/161/161，
D为block112/141/141/141；不要求整个作业位于同一domain。
四组manifest去掉topology后均严格等于各自恢复checkpoint的contract。
B/C/D首个loss更新均为1194，input_sha256均为
`e291db798d9d3a7cd9d5010a1a34d0ceda07c2e59679ba70140458e1afa0cd82`，
LR均为0.00027606561431923283，loss/grad_norm有限；C已推进1199。
A首个1594更新及跨所有rank输入日志的核验仍待完成，不能据此宣布50B完成。

2026-09-19 19:16 UTC，v12 按空间压力策略再次清理：清单
`runtime/mor-pretraining/formal-driver-v1/retention-1789870593109970863.json`
及同名 `.log` 确认删除 A 的 step 1300/1400 共 32 个 distcp，
855,058,915,858 bytes；这些权重/optimizer 分片不可恢复，未删除数据或日志。
清理后控制器记录可用 6,461,094,277,120 bytes，高于预算
5,924,292,859,280 bytes。独立远端只读核查两处 distcp 均已不存在；
A1500/1593 和四组1193各保留16个非空distcp、非空.metadata及完成标记。
此检查仅证明文件结构，不是全tensor校验。source58/bundle08901及训练配置不变；
7292218/7289637/7289665/7289784仍PENDING，控制器701996存活。

2026-09-19 14:40 UTC 调度只读核查：7292218/7289637/7289665/7289784
均 PENDING、Dependency=(null)、64 GPU、16 节点、SegmentSize=4、4 小时时限，
无 restart；不是 ABCD 之间的依赖阻塞。B/C/D 的 LastSchedEval 仍为 12:12:30。
同期 sdiag 显示调度器持续工作（统计起点 13:37:36 后启动 478 个作业），
全局 pending=3666、running=658，30 次 backfill 中 29 次达到 bf_max_time。
这些是调度压力的证据，不能据此确定单个作业的唯一排队原因或启动时间；
未修改优先级、资源、segment 或重提任务，继续监控现有作业。

2026-09-19 A7289602的1193→1593段已由sacct确认COMPLETED、ExitCode=0:0，
elapsed=02:15:52（各step亦0:0）。累计13,363,052,544输入tokens，固定验证集
50,753,524 targets：NLL=2.7624075504345633、PPL=15.837927694093825，
policy=causal-full-depth；本段cost=142.97657354551058 GPU-hours，非累计成本。
1593 checkpoint完成标记、16个非空distcp（427,529,457,929 bytes）、64份
非空RNG、.metadata及合同/游标检查通过，非全tensor校验。配置仍为source58/
bundle08901、64 GPU、MBS2、EP16、segment4。随后控制器已审计7289602，completed.A
更新为1593，本地evidence/7289602收齐400次连续有限更新、64份rank输入日志及
段末评估。下一段1593→1993（16,718,495,744累计输入tokens）提交子进程877152
已确认存活，恢复源checkpoint-001593；随后自动提交成功：Slurm7292218、registry
20260919-143101-3424。独立scontrol确认64 GPU、16节点、SegmentSize=4、
TimeLimit=04:00:00、Dependency=(null)，当前PENDING；v12回到watching，无error/intent。
未重复提交，启动后的恢复首步及EP拓扑仍待核验。
B/C/D的1193→1593段仍排队，不能将A中间结果称作四组最终结果。

2026-09-19 A7289602的1500步保存通过文件结构/合同检查（source58/bundle08901，
64 GPU、MBS2、EP16、segment4）：16个非空distcp合计427,529,457,929 bytes，
64份非空RNG及非空.metadata；完成标记step=1500、cursor=12,582,912,000，
合同与本段manifest除topology外一致。随后1194→1503更新连续，LM loss、
grad_norm、LR均有限。非全tensor校验或新进程恢复测试。B/C/D下一段仍PENDING，
本段1593及最终5954步均尚未完成。

2026-09-19 v12压力清理首次实际触发（用户已授权空间不足时清理冗余checkpoint）：
清单`runtime/mor-pretraining/formal-driver-v1/retention-1789850222306267894.json`
及同名.log记录272个distcp删除，共5,069,212,950,769 bytes。对应A的
500/600/700/800/900/1000/1100/1200、B的500/600/700/800/900/1000、
D的900/920/1000，共17个冗余保存点；C未删。删除权重/optimizer分片不可恢复，
不是删除日志、数据或RNG/metadata。独立远端只读核查272个目标均不存在，
选取的13个保护点（四组120/1193、B/C/D1100、A1300/1400）均仍有16份非空
distcp、64份非空RNG、完成标记及.metadata。清理后可用9,324,813,996,032
bytes，高于预算4,513,445,648,114 bytes；A7289602仍RUNNING至1447，
B7289637/C7289665/D7289784仍PENDING。source58/bundle08901及训练配置未改。

2026-09-19 A7289602的1400步保存亦通过相同检查：16个非空distcp共
427,529,457,929 bytes、64份非空RNG及非空.metadata，完成标记step=1400、
cursor=11,744,051,200；合同与本段manifest除topology外一致。
保存后1194→1403更新连续，LM loss、grad_norm、LR均有限。仍仅为结构/合同检查，
非全tensor校验；B/C/D下一段仍PENDING，四组5954步目标未完成。

2026-09-19 A7289602已通过1300步保存检查（source58/bundle08901，配置沿用下文）：
完成标记step=1300、cursor=10,905,190,400，合同与本段manifest除topology外一致。
16个非空distcp共427,529,457,929 bytes，64份非空RNG及非空.metadata；
随后训练已继续到1302步。本次为文件结构与合同检查，不是全tensor校验或新进程恢复测试。
B7289637/C7289665/D7289784仍PENDING。v12空间检查通过、removed_bytes=0，
无需修改训练配置或删除checkpoint；四组5954步目标尚未完成。

A7289602恢复后已从1194连续更新到1200，数值有限，64份rank输入日志首步
均为1194。独立核对64份拓扑记录：四个EP group分别位于block021/021/013/013，
每组16 ranks、4节点且单NVL domain，成员声明一致。新manifest除topology外
与恢复源1193 checkpoint合同完全一致。1200保存完成：16个非空distcp合计
427,529,457,929 bytes、64份非空RNG及.metadata，游标10,066,329,600及
合同正确；随后观察到1202更新。该检查不是全tensor校验；B/C/D仍待启动核验。

10B之后自动续跑已实际提交：A7289602、B7289637、C7289665、D7289784，
均从1193续跑至1593步（累计13,363,052,544输入tokens）。逐个scontrol确认
64 GPU、16节点、SegmentSize=4、TimeLimit=04:00:00、Dependency=(null)，
MBS分别2/2/4/2；当前均PENDING，尚未证明恢复后首步更新通过。
v12控制器已回到watching，四组均有registry/scheduler记录，无待提交intent或error。
此次无需手动重提、改变训练配方或删除checkpoint；后续继续核对启动、EP domain、
恢复游标/首步输入及1593段末审计，最终目标仍为各组5954步。

2026-09-19 D组10B阶段：作业7281040（64 GPU、MBS2、EP16、segment4，
source58/bundle08901及固定环境）已由sacct确认COMPLETED、ExitCode=0:0，
耗时01:57:18。1193步累计10,007,609,344输入tokens；固定验证集50,753,524
targets的NLL=2.919834552680463、PPL=18.538220107185516，policy=
`strict-causal-threshold`，实际平均递归深度1.9442008722426556。
评估估算forward FLOPs=2.698409626549125e17（不是实测硬件FLOPs）；
本段cost=122.26915859181848 GPU-hours，并非累计成本。
checkpoint-001193有完成标记、16个非空distcp（183,219,768,553 bytes）、
64份非空rank RNG及.metadata，步数/游标/合同检查通过（非全tensor校验）。
四组同边界NLL：A=2.9067436149210892、B=2.9631551365756086、
C=2.955054697466526、D=2.919834552680463；D−B=-0.0433205838951456。
仅为单seed的10B中间检查点结果，不能替代最终50B结论。随后控制器已完成
7281040审计，四组completed均为1193；本地evidence/7281040收齐273次连续
有限更新和64份rank输入日志及评估等段末证据。控制器已开始A1193→1593
的提交子进程，后续四组实际作业编号仍待确认，未提前宣称完整实验完成。

2026-09-19 A组10B阶段：作业7287488（64 GPU、MBS2、EP16、segment4，
source58/bundle08901及固定环境）已由sacct确认COMPLETED、ExitCode=0:0，
耗时01:12:19。1193步累计10,007,609,344输入tokens，固定验证集50,753,524
targets的NLL=2.9067436149210892、PPL=18.29711898465554，policy=
`causal-full-depth`。本段cost为74.31461764931679 GPU-hours，并非累计成本。
checkpoint-001193有完成标记、16个非空distcp（427,529,457,929 bytes）、
64份非空rank RNG及.metadata，1193步/游标/合同检查通过（非全tensor校验）。
同边界B−A NLL为+0.0564115216545194；A已低于C，与早期阶段排序不同，
仍须等待D及最终单遍训练结果，不据单个中间检查点外推效果。随后控制器已完成
7287488审计、completed.A=1193，本地evidence/7287488收齐193次连续有限更新、
64份rank输入日志及评估等段末证据；A/B/C均已通过10B段末审计，等待D。

2026-09-19 B组10B阶段：作业7287446（64 GPU、MBS2、EP16、segment4，
source58/bundle08901及本文所列固定环境）已由sacct确认COMPLETED、ExitCode=0:0，
耗时01:06:50。1193步累计10,007,609,344输入tokens；固定验证集50,753,524
targets的NLL=2.9631551365756086，PPL=19.35895564768735，policy=
`causal-full-depth`。本段cost为68.46645395971835 GPU-hours，并非累计成本。
checkpoint-001193有完成标记、16个非空distcp（183,219,663,613 bytes）、
64份非空rank RNG和.metadata；游标、scheduler及去掉topology后的manifest合同一致。
此处为产物布局/合同检查，非全tensor校验。随后控制器已完成7287446段末审计，
completed.B=1193；本地evidence/7287446已收齐193次连续且数值有限的更新、
64份rank输入日志、评估/cost/complete及checkpoint状态。A/D同边界结果仍待确认。
同token的C组NLL为2.955054697466526，B−C=+0.0081004391090826；
仅为单seed中间检查点结果，不代表最终50B结论。原四组10B屏障保持不变。

本次分段定期保存核验：A1100、B1100、D1000均有完成标记，分别16个distcp、
64份非空rank RNG及非空.metadata；分片总字节分别427,529,457,929、
183,219,663,613、183,219,768,553。步数/游标正确（1100为9,227,468,800、
1000为8,388,608,000），scheduler=`pure-token-function-v1`，训练合同与新manifest
去掉额外topology字段后完全一致。该字段按train.py设计只加入manifest，不属于
checkpoint合同；临时检查最初将两种结构直接比较导致误报，未发现实际合同漂移。
观察到B1110、D1006已在保存后继续更新。这里只核验布局/合同/游标，不冒充全tensor
校验或此保存点的外部进程恢复测试；1193段末与50B仍待完成。

三组本次恢复后的首批实测更新：A7287488从1001到1005、B7287446从1001到1007、
D7281040从921到922，所读记录的LM loss/grad norm/LR均为有限值。
逐组比较新manifest与恢复checkpoint的contract，除实际节点topology外完全一致。
独立遍历64份rank记录及4个EP group确认：每组16ranks/4hosts/单NVL domain，
成员声明一致；A为block140/134/109/053，B为block064/165/099/035，
D为block007/037/070/049。此证据证明恢复后已有更新及布局合法，不替代1193步
段末完整审计或全量50B验收。

首批连续性补充核验：A1001–1009、B1001–1011、D921–925均连续且数值有限；
每组64份input-rank日志的首条步号分别为1001/1001/921。将三组已观察到的
各步与已完成C920→1193的原始loss记录比较，input_sha256、tokens、lr逐步
完全一致（分别9/11/5次更新）；不同组LM loss不要求一致，不以即时batch loss
替代固定验证集模型比较。

后续调度更新：A7287488、B7287446、D7281040均已从PENDING进入RUNNING，
三组分别获得16节点。初期日志仍为Python/CUDA依赖初始化警告，尚未据此认定
恢复或首步更新通过；须继续核对A/B从1000、D从920恢复及实际首批输入/路由证据。
此次状态变化未重提、未重启作业，控制器仍为v12条件清理模式。

当前控制器为v12（PID701996，`driver-pressure-host-v12.log`）：按用户后续明确授权，
启用`prune_on_pressure=True`，仅空间预算不足才清理保留策略之外的旧权重。
194项CPU回归及ruff通过；正常启动重审后state为watching，A/B/D作业编号不变，
没有重启GPU作业。首轮真实预算：可用9,240,313,393,152 bytes，当前各活跃段
所需2,498,029,117,503 bytes，准入通过，实际删除0 bytes。
下方retain-all及审核阻塞记录均为历史过程，不代表当前控制器状态。

A1000→1193后续已实际提交为7287488（registry `20260919-090735-f8d8`），
控制器记录submitted，不再只有intent。B7287446实查SLURM为64 GPU、16节点、
SegmentSize=4、4小时、无dependency，当前PENDING；原D7281040仍PENDING。
未改动科学配置或重复提交已有作业。

2026-09-19 16:06 UTC进展：A7280872和B7280773的600→1000段均已通过
控制器段末审计，累计8,388,608,000 tokens。相同50,753,524个有效验证targets上，
A NLL=3.0252295777119267、PPL=20.59873317843161；
B NLL=3.0809578926158436、PPL=21.779254594673997。
B1000→1193实际提交为7287446（registry `20260919-090547-709a`），从完整
B1000恢复；A下一段处于submitting，尚未记录确认ID。当前仍是保留全部checkpoint
模式，removed_bytes=0；C1193、D920，不能称为四组10B或50B完成。

最新恢复记录：不删除模式v11（PID694492，`driver-retain-all-host-v11.log`）
已通过原恢复与历史审计，state进入watching；B7280773的1000步段末审计通过。
实际空间检查可用9,240,331,857,920 bytes、当时预算1,154,381,050,112 bytes，
removed_bytes=0。B1000→1193已进入提交流程，尚不能仅凭intent认定已提交成功。
用户随后明确授权“当空间不足时”删除多余checkpoint；此为有条件授权，
不是允许提前定期删除。保留里程碑、各组最新两个完整点、活跃恢复源及所有
日志/metadata/RNG，删除前重新核验清单与空间缺口。当前控制器仍为retain-all模式，
并未声称条件清理已自动部署；下方此前安全阻塞说明为历史记录。

最新运行控制说明：v9控制器因checkpoint清理SSH认证前断连退出；修复已通过
187项CPU测试及远端只读核验。恢复v10的启动请求被安全审核拒绝，尚未执行，
需要用户明确批准恢复自动控制器及其过期checkpoint清理副作用。现有A/B GPU
作业仍在训练，D仍排队；不应把下文历史快照中的“控制器存活”当作当前状态。

2026-09-19 后续只读核验：A（7280872）已到737步，B（7280773）已到747步，
两者从601步起日志连续，LM loss、grad norm、LR均为有限值；重叠的601–737步
共137次更新，其输入hash、消费token数和LR逐步完全一致。两组700步checkpoint
均有完成标记。D（7281040）仍为PENDING。这证明现有训练继续推进，不代表控制器
已恢复，也不代表50B训练完成。用户回复“允许”后恢复启动再次被安全审核拒绝，
v10仍未执行；未绕过审核，也未因本次只读检查删除权重或提交新GPU作业。

状态截至2026-09-19 13:53 UTC：**C已完成10B并通过段末审计；A/B的600步
（5.033B）checkpoint已实际恢复，两组均观察到606步；D完成920步（7.718B），下一段仍排队。
四组全量约50B尚未完成。**
本文只记录启动结果，不把调优loss当作正式模型效果。调查细节另见
[设计与调查记录](pretraining-design.zh.md)。

| 组别 | 当前作业/最近完成作业 | 当前状态 | 最近已审计的验证结果 |
|---|---|---|---|
| A | 7280872，600→1000 | RUNNING，已核实606步，64 GPU、segment4、4h | 600步：NLL 3.42450002，PPL 30.70728798 |
| B | 7280773，600→1000 | RUNNING，已核实606步，64 GPU、segment4、4h | 600步：NLL 3.48867734，PPL 32.74261161 |
| C | 7279982，920→1193 | COMPLETED 0:0，等待其余组到达10B屏障 | 1193步：NLL 2.95505470，PPL 19.20277303 |
| D | 7281040，920→1193 | PENDING，64 GPU、segment4、4h | 920步：NLL 3.05564418，PPL 21.23486026 |

上述NLL对应的训练token边界不同，不可横向作为等token效果比较。
C1193步累计10,007,609,344输入tokens；A/B600步累计5,033,164,800。
固定验证集均50,753,524个有效targets。D920步累计7,717,519,360输入tokens，
严格因果评估平均递归次数1.92022713。控制器v9（宿主PID473355）已实查存活，
日志`runtime/mor-pretraining/formal-driver-v1/driver-retention-host-v9.log`；
从完整checkpoint续跑，不重新初始化或重设学习率日程。

13:44 UTC启动快照：A/B的SLURM状态均为RUNNING；启动日志已执行容器内
训练命令，分别指定各自200→600段的`checkpoint-000600`、MBS2、64rank，
使用原bundle `08901c997f8058ab6e33f52d1d585aaee4fc668ee6ba847ed3e527b0747d0b69`。
此时尚未观察到601步更新，不把获得节点或resume命令当作实际恢复成功证明；
仍需检查64rank首步、恢复合同、输入hash/LR及EP domain布局。

13:53 UTC实际恢复核验：A/B各64rank的首条输入记录均为601步，当前manifest
除新节点topology外与各自600步checkpoint的contract完全一致；601–606步的
全局输入hash、tokens、LR两组逐项一致，已读更新的LM loss和grad_norm均有限。
四个EP16组均为4节点、单NVLink domain：A依次block100/100/100/150，
B依次block124/124/026/026。多个EP组可以落在同一domain；未要求domain互异。
这证明实际恢复与首批更新通过上述检查，不代表1000步段末或50B最终验收完成。

14:26 UTC中途保存核验：A7280872/B7280773均已写出700步checkpoint完成标记，
游标均为5,872,025,600，scheduler为`pure-token-function-v1`；各含16个distcp、
1份`.metadata`及64份RNG文件。A分片合计427,529,457,929 bytes，
B合计183,219,663,613 bytes；B已继续到703步。此处仅核验文件清单、大小与状态，
未声称对700步全部tensor做校验或已用700步进行新进程恢复。

## 早期续训记录（历史快照，非当前状态）

C `7243558`（48m10s）与D `7243582`（2h41m47s）均为COMPLETED、0:0。
各121–520共400步的序号、token/LR、有限loss/梯度、全rank显存、checkpoint及
最终评估通过正式审计，全部400步全局输入hash逐步一致。两组累计输入tokens均
4,362,076,160；固定验证有效targets均50,753,524。

| 组别 | 520步验证NLL | PPL | 验证平均递归次数 |
|---|---:|---:|---:|
| C | 3.53748245 | 34.38025619 | 不递归 |
| D | 3.48074529 | 32.48392288 | 1.86840841 |

这是同一中途更新边界的正式结果，不是10B/最终结果，也不能替代计划中的B/A、
B/C和D/B比较。A/B第200步通过只读恢复预检查，随后实际GPU恢复亦已确认；
A第300步不完整，不用于恢复。已按授权删除完成调优及过期正式点的权重分片，
保留里程碑、最新两个完整点和日志/配置/元数据；删除的权重不可直接恢复。
正式state经完整审计已恢复为A200/B200/C520/D520。
一次提交前SSH握手故障已确认未产生新作业，修复受限传输重试并归档失败intent后，
控制器v4继续提交；不会盲目重试可能已执行的sbatch。

v4成功提交A/B后，C遇到可证明未认证/exec的SSH握手失败；两端registry与
输出目录审计排除重复提交后恢复v5（宿主PID352158），C/D提交成功。

| 组别 | 续训作业 | 起点→段末 | MBS | 当前状态 |
|---|---|---|---:|---|
| A | 7278726 | 200→600 | 2 | RUNNING，最近已核实254 |
| B | 7278761 | 200→600 | 2 | RUNNING，最近已核实232 |
| C | 7279034 | 520→920 | 4 | RUNNING，最近已核实610 |
| D | 7279077 | 520→920 | 2 | RUNNING，已到522 |

四条实际registry命令的arm、resume、MBS、64GPU、stop_tokens/output均已核对；
四个SLURM记录均64GPU/16节点、SegmentSize4、4h、Dependency=null。
这是续训分段，不重新初始化、不重设学习率日程。controller状态watching，
日志`runtime/mor-pretraining/formal-driver-v1/driver-retention-host-v5.log`。

A恢复实证：64rank均从201步接续，201–203全局输入hash匹配原尝试，
checkpoint合同不变、token/LR连续、loss及梯度有限；四个EP group各位于
单一NVLink domain（block167/104/067/118）。203步训练LM loss5.52346，
不是固定验证集NLL。后续checkpoint保存和分段结束仍待监控审计。

B/C分别从201/521步接续，64rank首步一致、原checkpoint合同一致，
恢复后的已检查更新loss/grad_norm有限。B201–204输入hash及LR与A同一步一致；
B的四组EP位于block049/169/050/102，C位于block101/037/122/088，
每组16rank、4节点、单domain。此处进度不是分段完成或新checkpoint保存证明。

D恢复后64rank从521步接续，521–522输入hash/LR匹配C，合同不变，
loss/depth_aux/梯度有限；EP分别位于block167/016/067/174，未跨domain。
C第600步保存包含16个distcp和完整标记，随后继续训练；未将该文件检查冒充
逐tensor内容校验或该checkpoint的新进程恢复验证。四组均继续向10B/全量推进。

## 最终MBS

统一64×GB200、TP=CP=PP=ETP=1、EP16、segment4、BF16、seq4096、GBS2048。
吞吐为20次warmup之后30次更新的总输入tokens/总更新时间；整卡峰值为全部64
rank中的最大值，覆盖初始化、训练、初末验证和保存checkpoint。

| 组别 | 结构 | 独立参数 | MBS | 梯度累积 | 输入tokens/s | 整卡峰值 | 通过作业 |
|---|---|---:|---:|---:|---:|---:|---|
| A | 48层普通模型 | 30,532,122,624 | 2 | 16 | 366,359 | 73.06% | 7213308 |
| B | 3+14×3+3固定递归 | 13,084,744,704 | 2 | 16 | 364,174 | 70.74% | 7214574 |
| C | 3+14+3普通浅层 | 13,084,744,704 | 4 | 8 | 1,081,408 | 81.28% | 7215961 |
| D | B backbone加动态MoR | 13,084,750,848 | 2 | 16 | 376,492 | 57.83% | 7217234 |

独立审计再次读取全部11项最终数据扫描记录及A的2项pilot筛选记录，确认各组
`next_candidate=None`，选择结果与上表一致。所有成功档位均完成50更新、完整
验证和checkpoint；跨组/跨MBS的50个全局输入hash一致。

A/B/D的MBS4发生真实CUDA OOM，按失败档位保留原始证据；没有伪造吞吐或显存
峰值。C/MBS4峰值超过80%但低于90%，按既定规则合格并停止增大，不再测试8。
A已复用通过合同审计的pilot筛选，从最终数据MBS2复验；不重复从1扫描。

整卡显存以20ms采样，可能漏掉极短外部分配峰值。吞吐是短跑实测，不是长期
稳定吞吐承诺；D的动态路由计算量会随训练变化。所有调优token均不计入正式训练。

## 启动前验收证据

| 检查 | 范围与结果 | 作业 |
|---|---|---|
| 语义、因果和零活跃路径 | 32GPU小宽度四组，128份rank报告通过；前缀预测最大差值0 | 7217539 |
| 初始化与参数计数 | 64GPU官方完整宽度四组，256份报告通过；B/C/D初始backbone bitwise一致 | 7219258 |
| 共享梯度 | 共享参数梯度对照同native未共享展开的梯度和，32rank通过 | 7217889 |
| auxiliary与optimizer分片 | 四组共128份报告通过；FP64归约参考最大relative-L2约6.44e-9；正常启动，无debugger | 7219512 |
| A/B跨进程恢复 | 完整宽度、各64GPU、MBS2，128份报告通过 | 7219391 |
| C跨进程恢复 | 完整宽度、64GPU、MBS4，64份报告通过 | 7219612 |
| D跨进程恢复 | 完整宽度、64GPU、MBS2，64份报告通过 | 7219752 |
| 汇总与全数据hash | CPU核验全部证据、来源与文件hash后生成四组正式凭据 | 7220068 |

恢复检查包括模型、FP32 master、Adam状态、RNG、optimizer LR、数据游标以及
恢复后下一更新与不中断参考的bitwise一致性。因果语义证明来自小宽度模型，
不将其描述为完整30B模型的全前缀数值证明。共享梯度对照使用同native内核，
不是独立内核精度对照。加入并行调度回归后预训练相关CPU测试110项通过。

## 数据与版本

- 原始文本镜像：`gmongaras/SlimPajama-627B_Reupload`，revision
  `c34c22dbb10ae6b264a2f357a909d1a537141b36`；不声称独立证明与不可访问的原仓库字节一致。
- 官方tokenizer：`Qwen/Qwen3-30B-A3B-Base`，revision
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`，EOS151643，无chat template。
- 总存储50,000,007,401 tokens；train49,949,241,482，validation50,765,919。
  文档token内容稳定hash划分，四组共用shuffle和packing。验证有效targets50,753,524。
- 正式每组5954更新，消费49,945,772,032输入tokens；舍弃训练尾部3,469,450。
- 数据manifest SHA：`fbbeba8051302c86818f4edd8d95131fb61a579bbbfa5f0ed5d54def7e7d6153`。
- 生产Python SHA：`58b909e36cc88bea5ce1efc1e8f400c4f5597bd5384c1ba631019c72df62359a`。
- 源码/配置bundle：`08901c997f8058ab6e33f52d1d585aaee4fc668ee6ba847ed3e527b0747d0b69`。
- Megatron commit：`5c8315f12a64a7279eec58896af9e74ee3351b74`。
- 环境：固定`gb200-torch2603_231f7a2ae3a0.sqsh`，Torch
  `2.11.0a0+a6c236b9fd.nv26.03.46836102`、CUDA13.2、TE`2.14.0+f031cf8`；
  使用已验收的`gb200-env-nvrx060`及提前设定fused attention环境的Python wrapper。

## 正式运行与产物

### 最新异常：2026-09-18 14:40 UTC只读核查

120→520步段实际状态：A 7243509 FAILED（最后记录300步/2,516,582,400 tokens）；
B 7243525 FAILED（最后记录299步/2,508,193,792 tokens）；C 7243558 COMPLETED、
0:0（520步/4,362,076,160 tokens）；D 7243582 RUNNING（快照296步/
2,483,027,968 tokens）。这些是日志更新进度，不等于失败作业可恢复checkpoint游标。
C末次验证NLL=3.537482451614582、PPL=34.380256189021146；执行器遇A失败已停，
尚未由执行器完成C这一段的产物审计和下一段提交。

A在300步后checkpoint写出worker记录`[Errno 122] Disk quota exceeded`，并有
PyTorch序列化`unexpected pos`错误；B在`train.py:390`追加各rank输入记录时也报
`Disk quota exceeded`。因此两者不是仅凭退出143推测OOM或NCCL根因；已观察到
明确的写入配额错误。具体用户/组/项目配额、容量还是inode限额尚未检查，不声称
已有完整checkpoint可直接恢复。生产源码仍source58b909e3...、bundle08901c99...。
本轮没有删除任何数据或checkpoint，没有重提作业或跳过失败guard；需要先解决
存储配额、核验各组完整恢复点，再恢复自动推进。下方保留此前启动过程记录。

### 首段启动与并行切换记录

首作业7220271：A/MBS2、从随机初始化开始、`mode=train`、无resume，计划从
0到120更新，即1,006,632,960输入tokens。driver按四组先1B、再10B、最后一遍
推进；始终使用5954步完整token学习率日程。后续段间恢复完整状态，不重新初始化。

用户于9月18日明确批准各组独立排队并允许并发。当前正式作业如下；组间没有
Slurm依赖，同一组后续段只依赖自己的已审计checkpoint。仍保留四组到齐1B/10B
的阶段检查，最多四组同时使用256GPU。

| 组别 | 作业 | 02:05 UTC状态 | 当前阶段 |
|---|---|---|---|
| A | 7220271 | COMPLETED，0:0 | 已完成120步/1,006,632,960 tokens |
| B | 7237775 | RUNNING，启动阶段尚未观察到更新 | 0→120步 |
| C | 7241927 | PENDING | 0→120步 |
| D | 7242004 | PENDING | 0→120步 |

A的1B段耗时55m25s，末步训练LM loss=6.343400873243809，完整固定验证NLL=
6.3698369333371385、PPL=583.9625962950612，checkpoint及完成产物已通过原审计。
完整JSON证据位于`runtime/mor-pretraining/formal-driver-v1/evidence/7220271/`。
B于02:01:08 UTC由Backfill调起，排队3h11m54s，未取消或重提；C/D均无依赖。

实际于2026-09-17 21:51:20 UTC开始运行。启动manifest与已签发A凭据的实验、
源码、数据、环境和模型配置逐项一致；64rank的四个EP16组均通过单NVLink domain
及4节点×4GPU布局检查。完整初始验证NLL=12.386161704331121，targets=50,753,524。
第1步LM loss=12.384249215014279、LR=5.038629492777964e-06；首两步输入hash与
调优记录一致。已归档前8次成功optimizer更新，第8步消费67,108,864输入tokens，
LM loss=10.31216055387631。证据为
`runtime/mor-pretraining/formal-driver-v1/startup-evidence/7220271/`中的manifest、
evaluation和loss；这些启动结果不代表1B阶段已完成。

当前batch分区最长4小时，因此运行recipe仅把原24小时预约改为4小时，每段最多
400更新；首段120更新使用90分钟。B既有作业由scontrol调整时限、保留原提交时间；
C/D从生成的`formal-driver-v1/first-stage-recipe.yaml`提交，仅时限不同。
科学配置逐字段比较不变。
这只是调度切段，不改变总更新数，也不将不同结构的同token训练称为同FLOPs。

以下路径相对主toolkit根目录：

- 扫描结果：`runtime/mor-pretraining/finaldata-driver-v2/state.json`及`evidence/`。
- 四组凭据：`runtime/mor-pretraining/evidence/acceptance-source58-final-v1/{a,b,c,d}.json`；
  每份绑定1869项证据文件hash，由issuer生成，不手工设置通过项。
- 运行recipe：`runtime/mor-pretraining/formal-batch-4h.yaml`。
- driver状态：`runtime/mor-pretraining/formal-driver-v1/state.json`。
- 并行控制进程日志：`runtime/mor-pretraining/formal-driver-v1/driver-parallel-host-v2.log`；
  状态schema v2保存每组活跃作业，迁移前原state另行归档，A结果及B作业ID保留。
- 集群正式输出：`runtime/mor-pretraining/formal-source58-v1-a-000000-000120/`。
- 集群日志：`runtime/slurm/logs/mor-a-mbs2_7220271.log`。

当前只有A的正式1B结果，尚不能给四组配对结论，也没有10B或完整一遍结果。
最终结论限定为单seed1234、约50B
单遍、统一配方的语言建模效果；不能用上述性能短跑推断递归模型效果更好。
