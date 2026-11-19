# Firefly Y640 IK

使用 Pinocchio 和 SciPy 求解 Firefly Y640 双臂运动学。提供单臂 FK 和 IK、双臂 IK、关节限位及完整位姿分支枚举。调用时使用自己的 Firefly URDF。

## 安装

```bash
python -m pip install -e .
```

## 使用

将以下内容保存为 `ik.json`，并填入 URDF 路径。相对路径以配置文件所在目录为基准。可通过 `tcp_offsets` 指定各臂的法兰到 TCP 变换。

```json
{"urdf": "/path/to/firefly_dual.urdf"}
```

```bash
firefly-ik --config ik.json info
firefly-ik --config ik.json fk --arm left
firefly-ik --config ik.json validate --samples 20 --random-seed 42
```

用 `firefly-ik ik --help` 查看目标位姿、种子角和分支选项。位置单位为米，关节角和 RPY 单位为弧度，四元数顺序为 `[x,y,z,w]`。IK 的成功状态表示位姿残差和关节限位满足要求。碰撞和轨迹可执行性应由调用方验证。

Python API 可用 `FireflyIK.from_config("ik.json")` 创建求解器，调用 `fk`、`solve` 或 `solve_dual`。
