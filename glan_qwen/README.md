# Qwen3-Embedding-8B + GLAN / ZS-CSD

在 MT-CSD 根目录执行下列命令。训练使用 PPED 已有的冻结、目标条件化话语向量，
保留 `code/models/GLAN.py` 实际执行的图卷积、局部 CNN、全局交互、hop 更新和目标分支加权。
这是替换编码器后的 GLAN 适配版，不能称为原 BERT GLAN 的严格复现。

## 数据和标签

- JSON 与 `pped/datasets/dataset_csd_2_qwen_1.py` 一致：`id`、`target`、`sentences`、`speakers`、`label`，可选 `target_type`。
- **0=favor，1=against，2=neutral**，不是原 MT-CSD 的标签顺序。
- 直接复用上述 PPED 加载器的向量索引与完整性检查；只取当前对话（含最终话语）。
  历史标签、跨对话 speaker history、配对增强和 posterior 均不输入 GLAN。
- 支持全局 `embeddings.npy` + `sentence.csv(global_id,target,sentence)`，或各 split 的
  `<split>.npy` + `<split>_ids.npy`；后者要求所有历史话语在同 split 的最终话语中存在。
  若不存在则报错，不能用零向量或其他目标的向量代替。`--train-ids` 等可指定非默认 ID 文件。
- 训练前检查 train/valid/test 目标互斥；不改变原数据划分。

默认 YAML 已对齐 `csd/config/config_sitpcl_12_qwen_1.yaml`：读取
`qwen3_embedding_8b_target_final/train.npy`、`dev.npy`、`test.npy`，`sentence_csv: ''`，
并自动读取同目录的 `train_ids.npy`、`dev_ids.npy`、`test_ids.npy`。
不要混用 `qwen3_embedding_8b` 下未经核对的旧 CSV/向量；仅补列名无法改变向量的编码语义。
若 split 特征缺少历史话语，加载器会指出缺失项，此时需要完整的目标条件化话语缓存，不能跳过检查。

## 数据位置（PowerShell）

```powershell
data="/home/xiongjie/data/csd/data/raw_with_all_labels_v3"
features="/home/xiongjie/data/csd/data/raw_with_all_labels_v3/qwen3_embedding_8b"
```

MT-CSD/dataset 中是英文数据，不能作为这里的 ZS-CSD 输入。请将 $features 指向与论文 PPED 相同的真实特征缓存。

## 独立目标向量

原 GLAN 还单独编码目标文本，用于三个分支的加权。PPED 现有话语向量不包含这份独立存储。
请用**生成话语向量的同一 Qwen3-Embedding-8B 检查点**预计算一次：

```powershell
python glan_qwen/build_targets.py
```

脚本与训练共用 `config/config_glan_qwen.yaml`，也可仅传入另一份 YAML 路径。
输入使用 `train_path/dev_path/test_path`，输出使用 `target_store`。
`model_dir` 默认 `/home/xiongjie/data/models/qwen3-embedding-8b`；编码参数为
`embedding_batchsize`、`embedding_max_length`、`embedding_dtype`、`embedding_device_map`、
`embedding_normalize`、`trust_remote_code`，设备共用 `device/cuda_index`。
这些编码参数与 GLAN 训练的 batchsize 等参数相互独立。

此脚本复用现有预处理程序的模型加载逻辑，通过 transformers 编码原始目标文本，不加 instruction，左填充、最后 token 池化、默认 L2 归一化；
不读 stance 标签，不做训练。NPZ 包含唯一字符串数组 `targets` 与 `[T,D]` 的 `embeddings`。
可以合并三份 split 的目标文本建立冻结特征缓存；这不混合标签或拟合参数。
另存 metadata 记录模型和处理方式。需保证与现有话语向量的检查点、输出维度、池化及归一化设置一致；
如果原特征流水线有不同设置，应使用原流水线导出同一 NPZ 格式。

## 训练示例（全局话语向量存储）

```powershell
python glan_qwen/train.py --train-json $data/train_data.json --valid-json $data/dev_data.json --test-json $data/test_data.json --train-embeddings $features/embeddings.npy --valid-embeddings $features/embeddings.npy --test-embeddings $features/embeddings.npy --sentence-csv $features/sentence.csv --target-store glan_qwen/features/targets.npz --output-dir glan_qwen/runs/seed123 --seed 123 --device cuda
```

路径是模板；本机当前未发现上述 Qwen 特征文件，需要指向实际缓存。
默认：256 维共享线性投影 + LayerNorm，hop=3，lambdaa=0.2，dropout=0.2，
AdamW lr=3e-4 / weight_decay=1e-4，batch=64，最多 30 epochs，patience=10，梯度裁剪=1。
这些是可调整的适配设置，并非声称对 GLAN 已经调优。与 PPED 比较时请明确各自验证集调参预算。

按验证集 `f1_macro` 保存 best.pt，早停后恢复最佳模型，**仅此时评估测试集**，并重新评估 dev。
评估口径与 `csd/main_12_qwen_1.py` 的 `compute_metrics_sitpcl` / `_metrics_by_scope` 一致：

- 固定分组 `mixed`（所有样本）、`c`（claim）、`n`（noun phrase）；mixed 直接合并样本计算，不平均两组得分。
- 最终 metrics.json 使用原脚本同名的扁平键，例如 `test_mixed_f1_macro`、`test_c_f1_favor`、`dev_n_accuracy`。
  每组均输出 `f1_macro`、`f1_favor`、`f1_against`、`f1_neutral`、`f1_avg`、`accuracy`。
- 最终分组 `f1_avg` 是三类 F1 均值；训练期未带分组前缀的 `f1_avg` 与原训练回调一致，是 Favor/Against 两类均值。
- 严格保留原脚本边界行为：macro 按真实与预测标签中出现的类别计算；若不足三类，三个逐类 F1 和 f1_avg 均置零；空分组六个指标全为零。
- 指标为 0–1，论文百分数需乘 100。每个 seed 使用新的输出目录，防止覆盖。

输出 config.json、history.jsonl、metrics.json，以及最佳模型的 final_dev.csv、final_test.csv
（列与原脚本一致：doc_id、target_type、true、pred）。保留 test_predictions.csv 供现有下游流程使用。

## 与源码的差异

- 用冻结 Qwen 话语与目标向量替换 BERT；增加共享维度投影及输入 dropout。
- 原源码的 upper-diagonal 邻接方向照原样保留；动态支持任意对话长度，无固定 8 节点上限。
- 修复原始 `target[0]` 导致批内样本共享首个目标的问题，改用各自目标向量。
- 不复制原源码未参与 forward 的 RGCN、第三层 GCN、额外 LayerNorm，因其不影响函数且徒增依赖。
- 保留原始 hop 更新公式和全局 sigmoid 交互；未新增 speaker 边、注意力模块或标签监督。
- 不复用原始未初始化的 GraphConvolution 权重；使用 Xavier 权重和零偏置。

## 验证

```powershell
python glan_qwen/test_glan.py
```

检查 hop 更新与源码公式等价、单节点与超过 8 节点、批处理与逐样本一致、目标隔离、有限梯度、
历史标签不进入模型、缺失历史向量报错，以及完整的合成数据训练/最优模型恢复/测试输出流程。
指标测试从原脚本直接提取两个评估函数，比较正常分组、缺失类别、空分组和未知类型的结果，
并核对训练产出的 dev/test 预测文件与最终指标一致。该对照测试需保留相邻的 csd 项目目录。
未在真实 ZS-CSD 上训练，也未验证完整 8B 目标编码流程。建议正式实验跑多个随机种子并报告均值和标准差。

