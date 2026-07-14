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

- Python 3.8 或更高版本
- NumPy
- SciPy
- Numba

建议先创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy numba
```

如果项目副本中带有 `requirements.txt`，也可以直接使用：

```bash
python -m pip install -r requirements.txt
```

## 从源码运行

PxPore 可以直接从源码目录运行。在包含 `PxPore/` 源码文件夹的上一级目录中执行：

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m PxPore PxPore/sharing/single_H.gro \
  --grid 0.02 \
  --probe 0.0 \
  --threads 8 \
  --atoms PxPore/sharing/UFF.atoms \
  --pore \
  --stats
```

如果使用解压后的源码包，需要把直接包含 `PxPore` 包的目录加入 `PYTHONPATH`，例如：

```bash
export PYTHONPATH="/path/to/source_parent:$PYTHONPATH"
```

## 命令行使用

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
- `--atoms`：原子参数文件，用于覆盖默认半径和质量。格式为：
  `symbol Z mass(g/mol) LJsigma(nm) epsilon(K)`。
- `--threads`：Numba 线程数；`0` 表示使用可用线程数的一半。
- `--out_prefix`：输出文件前缀。
- `--no-surface`：禁用表面积分析。
- `--pore`：启用孔隙分析。
- `--porevis`：输出孔隙可视化结果。
- `--no-octree`：禁用八叉树细化。
- `--oct-level`：最大八叉树细化层数；默认值为 `4`。
- `--oct-grid`：最小八叉树叶节点尺寸，单位 nm；默认值为 `0.001`。
- `--cube`：输出 Gaussian cube 文件。
- `--cube-space`：cube 文件空间分辨率。
- `--smooth`：对输出场进行平滑。
- `--stats`：输出统计 JSON。
- `--debug`：保存中间数组。
- `--debug-print`：打印额外调试信息。

## 输出文件

根据所选参数，PxPore 会输出：

- 包含几何和孔隙描述符的统计 JSON 文件；
- 可选的体数据 cube 文件；
- 可选的孔隙可视化结果；
- 可选的中间调试数组。

## 引用

如果使用 PxPore，请引用相关论文或代码仓库记录。

## 许可证

公开发布时应同时提供许可证信息。
