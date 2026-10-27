# 发布清单（维护者向）

给未来的自己：每次发版按这张表走，别凭记忆。
打勾之前**不要**推 tag —— tag 推上去就删不干净了。

---

## 1. 代码闸门

- [ ] `pytest -q` **全绿**（0.10.0 基线：525 passed）
- [ ] `python scripts\verify_launcher.py` 通过（改了启动器参数才需要）
- [ ] `python -m handmouse --safe-mode --no-preview --diagnostic-seconds 6`
      能正常跑完（**必须带 `--safe-mode`**：不加会真的注入鼠标）
- [ ] `python scripts\probe_camera.py` 输出与预期一致（换过硬件才需要）
- [ ] 如果你维护了多份副本，确认它们一致（以你在跑的那份为权威）：
      ```bat
      diff -rq . "<另一份副本的路径>" ^
        -x logs -x gesture_data -x __pycache__ -x .git -x "*.bat" -x .pytest_cache
      ```
      应当**没有输出**。

## 1.5 打包 exe（Release 附件用）

```bat
python -m pip install pyinstaller
pyinstaller handmouse.spec --noconfirm --clean
```

**实测（2026-10-25，作者机器）**：`dist/HandSolo.exe` **117MB**，无控制台（`runw.exe`），
图标已嵌入；带预览窗跑 6 秒共 **11.8 秒**（含 onefile 解压 2~4 秒），
摄像头 640×480@30fps、MediaPipe 模型加载正常、退出码 0。

打包后**必须**做的四项冒烟（前三项我这边能做，第四项只能靠干净机器）：

- [ ] `HandSolo.exe --safe-mode --no-preview --diagnostic-seconds 6` → 退出码 0；
- [ ] **以 DETACHED_PROCESS / ShellExecute 启动**（等价于双击，没有标准句柄）→ 确认日志落到
      `logs\handmouse-<日期>.log`（有 `personal\` 时是 `personal\logs\`）且内容是干净 UTF-8；
- [ ] **配置路径要对**：日志里 `配置=` 指向你预期的那个 `config.json`。
      有 `personal\` 时必须是 `dist\personal\config.json`——指到 `dist\config.json`
      就说明 `personal/` 没生效，手感会退回出厂默认（这条最容易漏，2026-10-27 就是栽在这）；
- [ ] 在**没装 Python 的机器**上双击：启动 → 摄像头 → 预览窗 → 光标注入四条。

🔴 已经踩过、别再踩的五个坑（都是"源码跑得好、打包就崩"的类型）：

| 症状 | 真因 |
|---|---|
| `ModuleNotFoundError: No module named 'tkinter'` | `excludes` 里有 `tkinter`，但 `snap_ui.py` 顶层就 import 它（那个排除项是打包配置写早了） |
| 双击没反应、也找不到日志 | `__main__._ensure_output()` 自己拼 `__file__.parent.parent`，单文件 exe 里那是**一次性解压目录**，日志写进去退出即删 → 必须用 `config.PROJECT_ROOT` |
| 首次提醒让人 `python -m handmouse --onboarding` | exe 用户没有 Python → `_onboarding_hint()` 按 `sys.frozen` 给不同命令 |
| 双击后**什么都不弹**，用户不知道要设置 | `console=False` 下 `print` 进了日志，屏幕上一片安静 → 冻结版必须用 `_ask_run_onboarding()` 弹系统对话框 |
| exe 里**控件磁吸静默消失** | 依赖 uiautomation / comtypes 只在 `cursor_snap.py` 的函数体里 import。静态扫描通常能抓到，但构建环境一缺包就**静默降级**，不报错。→ `hiddenimports` 里显式列死 |

## 1.6 `dist/` 的两种形态（打包完必看）

`dist/` 同时承担两个角色，靠**有没有 `personal/` 子目录**区分：

| 形态 | 长什么样 | 谁用 |
|---|---|---|
| **发布基线** | `dist/HandSolo.exe` + `dist/config.json`（出厂默认值） | 压成 zip 挂 Release。**绝不能带个人数据** |
| **个人版** | 上面这些 + `dist/personal/`（`config.json`、`gesture_data/`、`logs/`） | 作者本机日常用，手感与源码版一致 |

- 发布前检查 `dist/config.json` **逐字段等于** `AppConfig()` 的默认值。
  最省事的做法是直接 `ConfigStore(Path("dist/config.json")).reset()`——
  只要跑过一次 exe，这个文件就会被写进 `onboarding_reminded: true`
  之类的运行痕迹，**后果是陌生人永远看不到"该设置一下"的提示**（已踩过一次）。
- 打包**不会**清掉 `dist/personal/`，但换机器 / 重新 clone 后要重跑
  `同步个人设置到exe版.bat`（内部调 `scripts/sync_personal.py`）。
- 🔴 打 zip 时**排除** `personal/`：那个目录装的是作者的手标定参数和 1203 帧个人样本。

## 2. 版本号（三处，缺一不可）

| 文件 | 改什么 |
|---|---|
| `handmouse/__init__.py` | `__version__ = "x.y.z"` |
| `pyproject.toml` | `version = "x.y.z"` |
| `CHANGELOG.md` | 新增一节，标题带日期 |

`tests/test_branding.py` 会守住前两处不漂移；产品名只从
`handmouse/__init__.py` 的 `APP_NAME` 取，**不要在任何地方硬写产品名**。

## 3. CHANGELOG 必须写"为什么"

只写"改了什么"没有价值 —— 半年后你需要知道的是**当时为什么这么改、
什么方案被否决过**。参考 0.10.0 那节的写法：

- 用户原话（一字不改地引）；
- 实测数字（改前 / 改后 / 结论）；
- 被否决的方案与原因（"❌ 别试……"）；
- **未验证的部分要如实标注**。

## 4. 文档同步（最容易漏）

- [ ] 新增功能 → `README.md` 的「手势一览」与「命令行」；
- [ ] **删除功能 → 立刻删掉 README 里的对应行**。
      0.10.0 重写 README 时才发现：滚轮模式 2026-10-19 就删了，
      README 里还写着「张开手掌 → 滚轮模式」，陌生人照做会做出一堆没反应的动作。
- [ ] **改过 `collect.STEPS`（手势集）→ 同时改三处**：`docs/first-run.md` 的练习步骤表、
      README 的手势表、以及 `tests/test_onboarding.py` 里"练习集覆盖同一种手势"的断言；
- [ ] 新增配置键 → `docs/tuning.md`（如果是调参键）或保持默认不提；
- [ ] 新增踩坑 → `docs/troubleshooting.md`。

## 5. 提交

```bat
cd "<你的项目目录>"

:: 第一次发布前先设身份（改成你自己的 GitHub 邮箱）
git config user.name  "你的名字"
git config user.email "你的邮箱"

git add -A
git status                     :: 看一眼有没有误加 logs/ 或 gesture_data/
git commit -m "release: 0.10.0 —— 摄像头/宽高比自适应 + 开源首发"
```

⚠️ 提交前确认 `.gitignore` 已挡住 `logs/`、`gesture_data/`、`__pycache__/`、
`config.json.bak-*`。`config.json` 本身**是**要提交的 —— 它是作者实机标定的
起点配置，对使用者有价值，且不含任何个人信息。

## 6. 发布到 GitHub（第一次）

```bat
:: 1) 在 github.com 上先建一个空仓库（不要勾 README / .gitignore / LICENSE）
:: 2) 关联远端并推上去
git remote add origin https://github.com/<你的用户名>/HandSolo.git
git branch -M main
git push -u origin main
```

之后每次发版：

```bat
git tag -a v0.10.0 -m "0.10.0 —— 摄像头/宽高比自适应 + 开源首发"
git push origin v0.10.0
```

然后在 GitHub 上 **Releases → Draft a new release**，选 `v0.10.0`，
标题写版本名，正文直接粘 `CHANGELOG.md` 里对应那一节。

## 7. 仓库设置建议

| 项目 | 设成什么 | 为什么 |
|---|---|---|
| Description | `为一只手校准的摄像头鼠标 · Windows` | 一句话说清是什么 |
| Topics | `hand-tracking` `mediapipe` `gesture-control` `mouse` `accessibility` `windows` `python` | 决定别人搜不搜得到 |
| License | MIT | 与 `LICENSE` 一致 |
| Issues | 开启 | 实机失败报告是最有价值的输入 |
| Discussions | 可选 | 调参交流比 issue 更合适 |

README 顶部若要加截图／演示 GIF，放在 `docs/` 下并在 README 里引用 ——
0.10.0 还没有演示 GIF，这是 1.0 的门槛之一。

## 8. 1.0 的门槛（目前缺什么）

- [ ] 多用户验证：3~5 个人各自调完阈值能用；
- [ ] 已知缺陷清零（见 `docs/known-issues.md`）；
- [ ] README 有演示 GIF；
- [ ] 拒识率写进 README（现在只有内部数字）；
- [ ] `pyinstaller` 在无 Python 的干净机器上冒烟通过。
