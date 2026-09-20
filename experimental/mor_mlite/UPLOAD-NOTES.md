# MoR-MLite source snapshot

本目录是 MoR-MLite 训练框架的源码快照，包含源码、测试、配置、启动脚本和实验文档；不包含数据集、checkpoint、日志或 runtime 产物。

- Megatron-LM 基础版本：`5c8315f12a64a7279eec58896af9e74ee3351b74`。本分支保留该版本，不宣称兼容最新 main/dev。
- 框架入口与安装说明：`README.md`。
- 四组预训练设计及运行说明：`docs/pretraining-design.zh.md`、`docs/pretraining-launch.zh.md`。
- 部分脚本及历史文档引用原集群的绝对路径，其他环境需要配置相应路径和依赖。历史证据路径不随本快照上传。
- 本快照不代表正式约 50B 训练已完成；历史文档及 README 中的阶段性结论应结合对应日期和配置理解。

在本 Megatron-LM checkout 使用 MLite 时，将仓库的 `experimental/lite` 加入 `PYTHONPATH`，然后按本目录 README 安装框架及相应 GPU 环境依赖。

发布检查说明：Megatron 的 `tools/autoformat.sh` 仅覆盖 `megatron/core` 和 `tests`，不覆盖本目录。额外执行的框架 Ruff 检查发现已有的 21 项风格问题（19 项测试文件 E402、2 项诊断脚本 E731）；为保留源码快照，未修改这些文件。

发布时 `CHECK_ONLY=true BASE_REF=dev bash tools/autoformat.sh` 通过，其覆盖范围内没有变更；版权检查的同一范围内也没有新增或修改的 Python 文件。框架 `tests/test_pretraining*.py` 的 197 项 CPU 测试通过，使用原有固定依赖 checkout 和本地 toolkit 环境。这不等价于重新完成 GPU 多节点验收。
