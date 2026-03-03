# OAK-D 到 Mesh 的 Python 程序

这个仓库提供了一个示例脚本 `oakd_to_mesh.py`，用于从 OAK-D 的 RGB + 深度数据生成点云和三角网格（mesh）。

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 运行

```bash
python oakd_to_mesh.py --out_dir outputs --mesh_method poisson --visualize
```

运行后会在 `outputs/` 下保存：
- RGB 图
- 深度可视化图
- 点云 (`.ply`)
- 网格 (`.ply`)

## 3. 常用参数

- `--min_depth_m` / `--max_depth_m`: 深度裁剪范围（米）
- `--voxel_size`: 点云体素降采样大小
- `--mesh_method`: `poisson`（更平滑）或 `ball_pivoting`（更保边缘）
- `--median`: 深度中值滤波核大小

## 4. 精度建议

1. 使用 OAK-D 官方标定参数（脚本默认从设备读取内参）。
2. 保证光照均匀，尽量减少反光材质。
3. 多视角采集并做位姿配准（可在此脚本基础上扩展为多帧融合）。
4. 对输出点云做离群点滤除、法线估计和网格后处理（脚本已包含基础版本）。
