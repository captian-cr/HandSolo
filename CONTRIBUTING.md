# 贡献指南

先说清楚这个项目的性质：**它是一个"为一只手校准"的个人产品**，不是一个
通用的开源框架。所以贡献的方式和普通项目不太一样 —— 最常见的贡献不是加功能，
而是**在你自己身上复现并报告一次失败**。

---

## 最有用的四类贡献

| 类型 | 具体做什么 | 价值 |
|---|---|---|
| **1. 实机失败报告** | 在你自己机器/你的手上跑一遍，把「什么姿势、什么症状、能不能稳定复现」写清楚 | ⭐⭐⭐ 最高 |
| **2. 硬件适配** | 跑 `scripts/probe_camera.py`，把你摄像头的原生档位/宽高比/帧率贴出来 | ⭐⭐⭐ |
| **3. 判据改进** | 改 `gestures.py` 里的判据，并**附上审计脚本的前后对比数字** | ⭐⭐ |
| **4. 文档修正** | 发现文档与实际行为不符（这个项目里出现过多次） | ⭐⭐ |

---

## 环境

```bat
:: 1) 建虚拟环境（Python >= 3.10，推荐 3.12）
python -m venv .venv
.venv\Scripts\activate

:: 2) 装依赖（core 就够，uia 用于磁吸，all 是全部）
pip install -e ".[core]"

:: 3) 跑测试
pytest -q
```

测试基线：**525 passed**。改动前后都跑一次，不要接受"只挂了一个不相关的用例"。

```bat
:: 想要干净结果先清字节码缓存（从别处复制来的 .pyc 会加载旧字节码）
for /d /r %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
```

---

## 三条铁律（改代码前必读）

### 铁律 1：模型票永远无权推翻几何尺子

手势判据是**几何优先**的。样本人脸库（`_model_label`）只能把限制放宽一点点，
例如"无名指/小指收拢"这一条；**它不能发起一次几何判据都没通过的手势**。
违反这条会导致幻影滑动与光标冻结（历史上出过两次实机事故）。

### 铁律 2：有米制 3D 坐标就只听 3D

`HandObservation.world_landmarks` 齐全时，2D 判据**完全不参与**（不是并联，是降级兜底）。
把 2D 并进来（`OR`）会把它的误触率原样带回来 —— 实测 13.2% / 12.1%。

### 铁律 3：同一个问题修三次还不好，就换方案

不要打第四个补丁。项目里滚轮模式就是这么被整体删除的：
判据在侧手投影下根本不可分，任何阈值都救不回来。

---

## 改判据的正确姿势

1. **先跑审计脚本拿基线**（都在 `scripts/audit_*.py`，import 的是生产代码，
   所以不会与实现漂移）：
   ```bat
   python scripts\audit_swipe_final.py
   python scripts\audit_five_open.py
   python scripts\audit_dwell.py
   ```
2. 改判据；
3. 再跑同一个脚本，**把前后两组数字都贴进 PR**（可用度 / 各类误触发率）；
4. 补一个**直接覆盖该行**的测试。"相近的测试"不算 —— 这是踩过的坑。

没有数字的判据改动一律不接受，哪怕逻辑看起来更优雅。

---

## 不要做的事

| 别做 | 原因 |
|---|---|
| 加"自动切换采集后端"（`camera_backend` 之类） | 作者明确否决过：多一个自动分支就多一处不可复现的差异 |
| 在产品名/版本之外的地方硬写字符串 | 产品名单一来源是 `handmouse/__init__.py` 的 `APP_NAME` |
| 改包名 `handmouse` | 77 个文件有 import，留到 1.0 |
| 给后端类加公开方法但不改门面类 | `MouseController` 是门面，不转发会崩控制线程 |
| 提交 `logs/`、`gesture_data/`、`config.json` 的个人改动 | 已在 `.gitignore` / 属于本机状态 |
| 用 `print` 输出 | `pythonw` 下 `sys.stdout is None`，会 `AttributeError` |

---

## 提交信息

沿用 Conventional Commits，中文描述：

```
feat(gestures): 双指判据换用米制 3D 坐标
fix(snap): 磁吸窗口加释放宽限期，修相邻图标弹跳
docs(hardware): 补宽高比补偿的实测数据
test(vision): 覆盖驱动回报 0×0 的降级路径
```

正文里如果能附上"用户原话 + 实测数字"，比写一屏设计意图有用得多。

---

## 报告实机失败时请附

```bat
python scripts\probe_camera.py
python -m handmouse --safe-mode --diagnostic-seconds 20
type logs\handmouse-%date:~0,4%%date:~5,2%%date:~8,2%.log
```

加一句**能稳定复现的动作描述**（"食指朝下时捏合不触发，横向朝向正常"
远比"捏合有时候不好使"有用）。

---

## 许可

提交即表示同意以 [MIT](LICENSE) 分发。上游 `KIdiens/Vimoose` 的移植部分保留其
原始许可（`handmouse/LICENSE.vimoose`）。
