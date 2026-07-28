# PxPore parameter sensitivity study

本工作流仅使用仓库中的结构，通过 PxPore CLI 执行单因素扫描。脚本、
配置和说明统一位于 `scripts/sensitivity/`，唯一的中间文件目录为
`sensitivity_work/`，最终表格和图片写入
`docs/sensitivity_analysis/`。

绘图需要可选依赖：

```bash
python -m pip install -e '.[sensitivity]'
```

## 研究设计

| 扫描 | 变量 | 固定条件 | 主要输出 |
|---|---|---|---|
| Grid | `--grid` | `--no-octree`，surface samples=1000，PSD=centers | 体积、可达性、表面积、PLD/LCD、PSD |
| Octree | `--oct-level` | baseline grid，oct-grid=0.001 nm，surface samples=1000 | 体积、表面积、PLD/LCD |
| Surface | `--surface-samples` | baseline grid，`--no-octree`，关闭 pore | Sacc、Stotal |
| Probe | `--probe` | 仅 throat；grid=0.02 nm，关闭 octree/surface | PLD、LCD，以及原始 stats 中的 Vacc/Vtrap |
| PSD method | `--psd-method` | baseline grid，`--no-octree --no-surface --pore` | PSD descriptors、Wasserstein、JS distance |
| PSD local maximum | `--psd-local-max-mode` | centers，其余 PSD 参数取默认值 | center count、PSD descriptors、Wasserstein、JS distance |
| PSD minimum radius | `--psd-min-center-radius` | centers，其余 PSD 参数取默认值 | center count、PSD descriptors、Wasserstein、JS distance |
| PSD overlap pruning | `--psd-overlap-threshold`/`--no-psd-overlap-prune` | centers，其余 PSD 参数取默认值 | center count、PSD descriptors、Wasserstein、JS distance |
| PSD weighting | `--psd-hist-weighting` | centers，其余 PSD 参数取默认值 | PSD descriptors、Wasserstein、JS distance |
| PSD MC samples | `--psd-mc-samples` | MC，每个采样数使用 3 个 seed | PSD descriptors、Wasserstein、JS distance |
| Connectivity | `--transport-direction` | anisotropic_pore_x 和 anisotropic_sealed_x；periodic，grid=0.04 nm，关闭 octree/surface/pore | Vacc、Vtrap |

PSD 方法扫描使用一次 centers 和三次 MC。MC 固定 50000 samples，seed
为 `11451466/11451467/11451468`，seed 仅用于估算随机波动，不作为第五
个扫描参数。

PSD 专属参数范围为：

| 参数 | 范围 |
|---|---|
| local maximum | strict, plateau |
| minimum center radius/nm | 0, 0.005, 0.01, 0.02 |
| overlap threshold | 0.6, 0.75, 0.9, 1.0, 1.1, 1.25，以及关闭 pruning |
| histogram weighting | number, volume |
| MC samples | 10000, 50000, 200000；每项 3 个 seed |

## 结构和范围

| 体系 | Grid/nm | Octree | Surface samples | PSD |
|---|---|---|---|---|
| single_h | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1-5 | 100-10000 | centers/MC |
| h512 | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | centers/MC |
| overlapped_h | 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 | 1-5 | 100-10000 | centers/MC |
| throat | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | centers/MC |
| inaccessible_pocket | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-5 | 100-10000 | N/A |
| anisotropic_pore_x/z | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | centers/MC |
| anisotropic_sealed_x/z | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | N/A |
| HKUST1/IRMOF1 | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | centers/MC |
| ZIF8AP | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-3 at grid 0.02 nm | 100-10000 | centers/MC |
| equal/unequal overlapping, equal separated | 0.01, 0.015, 0.02, 0.03, 0.04 | 1-4 | 100-10000 | centers/MC 和全部 PSD 控制参数 |
| TMC-DAP | 0.02, 0.025, 0.03, 0.035, 0.04 | 1-3 at grid 0.04 nm | 100-10000 | centers/MC |

完整定义见 `scripts/sensitivity/sensitivity_study.json`。相同的
single-H 文件只使用
`tests/data/synthetic_systems/single_H.gro`；三种 MOF 统一使用 `.gro`，
不把输入格式验证混入参数敏感性。

三套双空腔 PSD 对照结构位于 `tests/data/synthetic_systems/`，脚本不生成
或修改这些数据文件。

各向异性体系的 grid、surface 和 PSD 扫描使用 periodic through-plane
连通性。当前 octree 连通性不支持 periodic directional，因此各向异性
octree 扫描固定使用 legacy/any，仅汇总 Vvoid、Vprobe 和 Stotal。较大
体系的 octree 深度上限和固定网格根据节点规模收紧，避免无物理收益的
数亿节点任务。

## 命令

只生成并检查任务矩阵，不运行计算：

```bash
python scripts/sensitivity/run_sensitivity_study.py --dry-run
```

执行全部扫描并生成结果：

```bash
python scripts/sensitivity/run_sensitivity_study.py
```

默认使用 8 个 CLI 进程，每个进程使用 8 个 OMP/Numba 线程。任务默认
断点续跑；仅当完整命令与缓存记录一致时才复用结果。

局部执行示例：

```bash
python scripts/sensitivity/run_sensitivity_study.py \
  --systems single_h,throat \
  --scans grid,surface
```

throat 非零探针验证：

```bash
python scripts/sensitivity/run_sensitivity_study.py \
  --systems throat --scans probe
```

扫描 `probe=0/0.05/0.10/0.14/0.20 nm`。理想几何参考为
`PLD=max(0, 0.46-2*probe)` 和
`LCD=max(0, 1.46-2*probe)`；当前扫描上限低于喉口关闭半径
`0.23 nm`。

仅重新执行并汇总全部 PSD 扫描：

```bash
python scripts/sensitivity/run_sensitivity_study.py --scans psd
```

需要完全重新计算时增加 `--no-resume`；默认会校验完整 CLI 命令后复用
匹配的缓存。

## 工作目录

`sensitivity_work/` 包含：

- `study_config.json`
- `run_matrix.json`
- `run_matrix.csv`
- `raw_results.csv`
- `runs/<run_id>/command.txt`
- `runs/<run_id>/stdout.log`
- `runs/<run_id>/stderr.log`
- 每次运行的 PxPore stats、center 和 PSD 输出

该目录包含运行时绝对路径、完整 CLI 命令和 stdout/stderr，仅用于本地
复现，已由 `.gitignore` 排除。

## 最终结果

`docs/sensitivity_analysis/tables/` 包含通用扫描结果：

- `study_design.csv/.md`
- `raw_results.csv`
- `metric_sensitivity.csv`
- `convergence_summary.csv/.md`
- `probe_sensitivity.csv/.md`

`docs/sensitivity_analysis/figures/` 为每个体系输出 grid、octree、surface
图片，并输出跨体系汇总图。

`docs/sensitivity_analysis/psd_method/` 单独保存全部 PSD 方法差异：

- `tables/psd_descriptors.csv`
- `tables/psd_parameter_sensitivity.csv`
- `tables/psd_method_comparison.csv/.md`
- `figures/<system>_<psd_scan>.png/.pdf`
- `figures/all_systems_psd_method_sensitivity.png/.pdf`

所有图片同时保存为 300 dpi PNG 和矢量 PDF。

`docs/sensitivity_analysis/figures/projections/` 包含 12 个解析、各向异性
及双空腔 PSD
测试体系的 XZ 中截面投影。灰色斜线表示原子占据区域，柔和蓝色色阶表示
到最近原子表面的距离；另提供 1×5 解析体系、1×4 各向异性体系和全部
测试体系总览，以及 1×3 双空腔 PSD 体系总览。
投影图同样同时输出 PNG 和 PDF。

解析体系使用理论值；没有解析真值的体系使用最细网格、最深 octree 或
最高表面积采样数作为数值参考。PSD centers 与 MC 只报告方法差异，不把
任一方法声明为真值。

通用敏感性图片中，体积、体积分数和表面积等非孔径标量使用双 Y 轴：
左轴为计算绝对值，右轴为相对参考值的有符号相对误差百分比。PLD、LCD、
PSD 和 center count 属于孔径/孔分布指标，仅绘制绝对值或分布曲线。
参考值为零时相对误差无定义，因此对应面板只绘制绝对值和参考值。

整个 `docs/sensitivity_analysis/` 同样属于可再生结果并由 `.gitignore`
排除。仓库只提交脚本、配置、说明和必要的合成结构，不提交本机路径、运行
日志、缓存、统计结果或图片。

已清理环境信息的原始数值结果快照位于
`results/sensitivity_analysis_raw_results.zip`，覆盖范围、缺失任务和
校验方法见 `results/README.md`。
