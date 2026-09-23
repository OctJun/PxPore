# PxPore

PxPore 是一个用于分子结构和分子动力学快照后处理的 Python 工具集，主要面向基于网格的孔隙分析、自由体积计算、可达/不可达体积分类、表面积估算和孔径描述符计算。

## 主要功能

- 读取正交晶胞的 `.gro`、`.xyz`、`.pdb` 和 `.cif` 结构文件。
- 计算晶胞体积、空隙体积、可达体积、不可达体积及对应体积分数。
- 估算可达表面积和总表面积。
- 计算 PLD、LCD 等孔径描述符。
- 支持在分子边界附近进行可选的八叉树细化。
- 使用 Numba 加速网格、连通性和孔隙分析核心计算。
- 可输出统计 JSON、Gaussian cube 文件和孔隙可视化结果。

## 环境要求

Python 3.10 或更高版本。当前源码版本为 `1.1.0`，运行依赖及版本约束以
`pyproject.toml` 为准（NumPy、SciPy、Numba、llvmlite、pandas、psutil、scikit-learn）。
在仓库根目录执行以下命令会同时安装包和所需依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 项目结构

```text
src/PxPore/        Python 包源码
src/PxPore/data/   包内数据，例如原子参数表
tests/data/        示例和检查用的输入结构文件
```

## 从源码运行

PxPore 可以直接从源码目录运行。在仓库根目录中执行：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p docs/example_run
cp tests/data/single_H.gro docs/example_run/input.gro
python -m PxPore docs/example_run/input.gro \
  --grid 0.02 \
  --probe 0.0 \
  --threads 8 \
  --atoms src/PxPore/data/UFF.atoms \
  --pore \
  --stats
```

如果使用解压后的源码包，需要把源码包里的 `src` 目录加入 `PYTHONPATH`，例如：

```bash
export PYTHONPATH="/path/to/source_tree/src:$PYTHONPATH"
```

## 命令行使用

PxPore 使用 Numba `njit` 内核加速主要计算。新的 Python 环境中第一次运行时，
这些内核需要先编译，因此首轮 wall time 可能包含一次性 JIT 编译开销。如果要
做性能测试或正式批量运行，建议先执行 warmup：

```bash
python -m PxPore.warmup
```

也可以先对任意一个代表性结构运行一次分析，再开始统计正式结果。

```bash
python -m PxPore input.gro \
  --grid 0.02 \
  --probe 0.0 \
  --threads 8 \
  --atoms UFF.atoms \
  --pore \
  --cube \
  --stats
```

该命令使用 0.02 nm 的网格间距和 0.0 nm 的探针半径分析 `input.gro`，启用孔隙分析并输出统计结果。可选的 `--cube` 参数会输出体数据 cube 文件。

## Python API

```python
from PxPore import analyse

result = analyse(
    input="structure.gro",
    grid=0.02,
    probe=0.0,
    atoms="UFF.atoms",
    threads=8,
    pore=True,
    stats=True,
)
```

## 参数说明

- `input`：输入结构文件。支持正交晶胞的 `.gro`、`.xyz`、`.pdb` 和 `.cif`。
- `--grid`, `-g`：目标网格间距，单位 nm；默认值为 `0.01`。
- `--probe`, `-p`：探针半径，单位 nm；默认值为 `0.0`。
- `--connectivity`：`legacy`（默认，非周期边界）或 `periodic`（周期贯通判定）。
- `--transport-direction`：可达性方向，`any`（默认）、`x`、`y` 或 `z`。
- `--atoms`：原子参数文件，用于覆盖默认半径和质量。格式为：
  `symbol Z mass(g/mol) LJsigma(nm) epsilon(K)`。
- `--threads`：Numba 线程数；`0` 表示使用可用线程数的一半。
- `--out_prefix`：输出文件前缀。
- `--no-surface`：禁用表面积分析。
- `--surface-samples`：每个原子的 Fibonacci 表面积采样点数；默认值为 `1000`。
- `--pore`：启用孔隙分析。
- `--porevis`：输出孔隙可视化结果。
- `--psd-method`：PSD 方法，可选 `mc`（默认）、`centers` 或 `both`。
- `--psd-mc-samples`：Monte Carlo PSD 采样数；默认值为 `50000`。
- `--psd-mc-seed`：Monte Carlo PSD 随机种子；默认值为 `11451466`。
- `--psd-mc-bin-size`：Monte Carlo PSD 的 bin 宽，单位 nm；默认使用网格间距。
- `--psd-mc-search`：最大包含球搜索实现，`pyramid`（默认）或 `offsets`（原始搜索实现）。
- `--psd-mc-grid`：MC 采样网格，`uniform`（默认）或 `octree`。
  `octree` 要求启用八叉树并使用 `pyramid` 搜索。
- `--psd-center-bin-size`：中心法 PSD 的 bin 宽，单位 nm；默认使用 `--grid`。
- `--psd-local-max-mode`：中心法局部极大值判据，`strict`（默认）或 `plateau`。
- `--psd-min-center-radius`：中心法最小球半径，默认 `0.005` nm。
- `--no-psd-overlap-prune`：关闭中心球重叠剪枝；默认启用剪枝。
- `--psd-overlap-threshold`：中心球重叠剪枝系数，默认 `1.0`。
- `--psd-hist-weighting`：中心法直方图权重，`volume`（默认）或 `number`。
- `--no-octree`：禁用八叉树细化。
- `--oct-level`：最大八叉树细化层数；默认值为 `2`。
- `--oct-grid`：最小八叉树叶节点尺寸，单位 nm；默认值为 `0.001`。
- `--cube`：输出 Gaussian cube 文件。
- `--cube-space`：cube 文件空间分辨率。
- `--smooth`：对输出场进行平滑。
- `--stats`：输出统计 JSON。
- `--debug`：保存中间数组。
- `--debug-print`：打印额外调试信息。

## PSD 与输出文件

`--pore` 才启用孔径计算；仅设置 PSD 参数不会启动计算。默认 MC 在可达的
均匀网格体素上采样，并为样本寻找最大包含球。体积/连通性仍默认启用八叉树；
`--psd-mc-grid uniform` 不等于 `--no-octree`。八叉树连通性支持周期边界及方向选择。
中心球坐标和中心法 PSD 需要显式指定 `--psd-method centers` 或 `both`。

输出写在输入文件所在目录。默认前缀为 `<输入文件名>_g_<grid>_p_<probe>`，
`--out_prefix` 可修改文件前缀。上面的源码运行示例先复制输入到 `docs/example_run/`。

| 文件后缀 | 启用条件 | 内容和单位 |
|---|---|---|
| `_stats.json` | `--stats` | 统计指标、设置及运行环境 |
| `_voxel_mc_psd.txt` | `--pore --stats`，MC/both | 7 列：编号、直径 nm、计数、概率、概率密度 nm⁻¹、PB 中心差分密度 nm⁻¹、累计概率 |
| `_Network-accessible_psd.txt` | 同上 | PB 格式的直径 Å、微分密度 Å⁻¹ |
| `_Network-accessible_psd_cumulative.txt` | 同上 | 探针直径 Å、剩余可达体积分数；随直径增大而递减 |
| `_psd.txt` | `--pore --stats`，centers/both | 中心法 PSD：编号、直径 nm、计数、权重份额、累计份额 |
| `_center.txt` | 同上 | 扩展 XYZ 格式；中心坐标 Å，球直径 nm |
| `*.cube` | `--cube` | 空隙、占据、可达、受困及距离场 |
| `_porevis.cube` | `--cube --pore --porevis`，centers/both | 中心球可视化 |
| `*.npy` / `*.npz` | `--debug` | 中间数组 |

PB 格式文件是 PxPore MC 结果的格式转换，不代表运行了 PoreBlazer。
默认 `mc` 不生成中心球；需要中心球可视化时使用 `both`。
Python API 返回统计、配置、网格、计时及输出路径信息；数值指标位于
`result["stats"]["stats"]`。无可达孔隙时，PLD/LCD 为 `-1`，PSD 主表为空。

可复现发布文件与当前源码的对应关系见 [docs/reproducibility.md](docs/reproducibility.md)。

## 敏感性分析

参数敏感性分析与绘图工作流见
[scripts/sensitivity/README.md](scripts/sensitivity/README.md)。清理本机路径
和运行环境信息后的原始数值结果包见
[results/README.md](results/README.md)。

## 绘图

脱敏后的论文绘图脚本、HMOF 对比 notebook 和结果数据包说明见
[scripts/plotting/README.md](scripts/plotting/README.md)。

## 引用

如果使用 PxPore，请引用相关论文或代码仓库记录。

## 许可证

PxPore 使用 MIT License 发布。详见 [LICENSE](LICENSE)。
