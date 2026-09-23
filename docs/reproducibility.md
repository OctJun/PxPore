# 可复现材料与源码对应关系

核对日期：2026-09-23。源码版本：`1.1.0`；此次文档核对基于
`a71cd5a6f5c619931a936cbc6b57ce23fce6655f`。发布时应记录实际归档的 Git
commit/tag；本文不表示 Zenodo 已上传或已分配 DOI。

## 源码行为

运行参数以 `src/PxPore/config.py`、`api.py` 和 `cli.py` 为准，计算与输出以
`core.py`、`pores.py` 为准。中英文 README 列出了当前参数和输出单位。

- PSD 默认 `mc`，50000 个样本，种子 `11451466`，搜索 `pyramid`，采样网格 `uniform`。
- 中心法及中心球可视化需要显式选择 `centers` 或 `both`。
- 八叉树默认启用，最大层数 2，最小叶尺寸 0.001 nm；它与 MC 采样网格是独立选项。
- `--psd-mc-grid octree` 要求启用八叉树并使用 `pyramid` 搜索。
- 连通性默认 `legacy/any`；源码同时支持周期方向连通性和八叉树混合连通性。
- MC 主表使用 nm 和 nm⁻¹；PoreBlazer 格式输出使用 Å 和 Å⁻¹。PB 累计曲线是剩余可达体积分数。

历史运行必须沿用其记录的参数和源码版本，不能因当前默认值改变而重新解释已有结果。
尤其应显式记录 PSD 方法、MC 搜索/采样网格、采样数、随机种子、bin 宽、八叉树和连通性设置。

## 发布材料清单

下面的两个压缩包已在作者工作区生成，位于工作区根目录的
`docs/r4_results_only_y1kjy6qv/`，并非 Git 仓库自带文件。
轨迹大小来自工作区 `docs/R4_reproducibility_audit/trajectory_sources.csv` 的历史核验记录。

| 材料 | 字节数 | 作用 |
|---|---:|---|
| `analysis_results_no_structures.tar.gz` | 67850425 | 新分析结果，不含结构 |
| `hmof_csv_and_plotting_scripts.tar.gz` | 148838603 | hMOF 表格及绘图脚本 |
| `APC-DAP/prod.xtc` | 219865620 | DAP 原始轨迹 |
| `APC-DAP/prod.tpr` | 5270612 | DAP 配套拓扑/运行输入 |
| `APC-MAP/prod.xtc` | 239694352 | MAP 原始轨迹 |
| `APC-MAP/prod.tpr` | 5609948 | MAP 配套拓扑/运行输入 |

共 6 个主要文件，687129560 字节（687.13 MB，655.30 MiB）。
其中两个现有压缩包共 216689028 字节。还需随附 README、SHA256 校验清单及实验映射。
两套轨迹应分目录存放，避免同名文件相互覆盖；上述大小不包含另行归档的源码和 Git 内 ZIP。

此方案依赖指定 Git 版本中的源码、`tests/data/` 和 `results/` 下已有 ZIP。
`results/fill_benchmark_results.zip` 保留第三方失败运行日志，不能用新结果包替换它。
两个补充性能输入已在 `a71cd5a` 提交到
[`docs/reproducibility_inputs/`](reproducibility_inputs/README.md)，无需单独上传另一份压缩包。

校验补充输入：

```bash
cd docs/reproducibility_inputs
sha256sum -c MANIFEST.sha256
```

hMOF CSV 支持结果重绘，不包含全量结构；从头重算还需要明确数据集版本、下载方式、ID 映射和预处理步骤。
旧 Git ZIP 与新结果包属于不同结果快照，绘图时必须明确选择哪一份数据，不应混用。

## 轨迹恢复

每个系统使用自己的 `prod.xtc` 和 `prod.tpr`。已核验的来源为作者工作区外
`Phenothalin-xlink/APC-DAP/data/` 与 `APC-MAP/data/`，不是系统顶层的同名较小轨迹。
以下命令要求当前目录含一套准确的 XTC+TPR，并使用新的输出目录。

209 帧时间序列（0–9984 ps，每 48 ps 一帧；DAP、MAP 分别运行）：

```bash
mkdir frames
printf 'System\n' | gmx trjconv -s prod.tpr -f prod.xtc -o frames/frame_.gro -sep -dt 48 -b 0 -e 9984
```

APC-DAP 11 帧对照（1000–2000 ps，每 100 ps 一帧）：

```bash
mkdir frames_11
printf 'System\n' | gmx trjconv -s prod.tpr -f prod.xtc -o frames_11/frame_.gro -sep -b 1000 -e 2000 -dt 100
```

历史 GROMACS 2024.4 验证记录显示，两套 209 帧共 418/418 个输入与实验输入
SHA256 一致，DAP 11 帧对照也逐字节一致；此次文档更新未重新抽帧。

## 尚需完成的发布工作

- 归档准确源码版本及上述文件，附每项校验和，并在发布后填写实际 DOI。
- 建立实验 → 输入 → 最新结果 → 重算命令 → 绘图命令的映射，提供目录恢复步骤。
- 验证恢复后的计算/绘图入口和 hMOF 外部数据获取步骤。

历史审核验证了输入恢复和部分绘图入口，未证明全部论文实验在新机器上端到端通过。
