# PxPore

PxPore 是一个用于分子动力学模拟后处理的 Python 工具集，主要用于计算孔隙结构、自由体积和孔径分布。

## 安装

假定已发布到 PyPI：

```bash
pip install pxpore
```

## 使用要求

- Python >= 3.8
- NumPy
- SciPy
- Numba (用于并行加速)
- 其他依赖：请查看 `requirements.txt`

## 使用案例

### 基本命令行使用

```bash
python -m PxPore input.gro --grid 0.01 --probe 0.0 --pore --cube --stats
```

这将分析 `input.gro` 文件，网格步长 0.01 nm，无探针，启用孔径分析，输出 .cube 文件和统计结果。

### Python API 使用

```python
from PxPore import analyse

config = {
    'input': 'structure.gro',
    'grid': 0.01,
    'probe': 0.1,
    'pore': True,
    'cube': True
}
result = analyse(**config)
```

## 参数说明

- `--input` / `-i`: 输入结构文件 (支持 .gro, .xyz, .pdb, .cif)
- `--grid` / `-g`: 网格步长 (nm)，默认 0.01
- `--probe` / `-p`: 探针半径 (nm)，默认 0.0
- `--atoms`: 原子信息文件，用于覆盖默认半径
- `--threads`: Numba 线程数，默认半数可用线程
- `--out_prefix`: 输出文件前缀
- `--no-surface`: 禁用表面积分析
- `--pore`: 启用孔径分析
- `--porevis`: 输出孔径可视化
- `--no-octree`: 禁用八叉树细化
- `--oct-level`: 八叉树最大层数，默认 4
- `--oct-grid`: 八叉树最小叶大小 (nm)，默认 0.001
- `--cube`: 输出 .cube 文件
- `--cube-space`: .cube 文件空间分辨率
- `--smooth`: 平滑输出
- `--stats`: 保存统计结果
- `--debug`: 保存中间数组
- `--debug-print`: 打印额外调试信息

## 引用

如果使用 PxPore，请正确引用：

[请提供相关论文或 GitHub 链接]

## 许可证

[请指定许可证]