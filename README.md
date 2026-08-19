# CPR-Coach 实时动作识别演示系统 · 实习交接文档

> 项目目标：复现 arXiv 2309.11718（CPR-Coach），基于 **ST-GCN 骨骼动作识别** + **YOLO 姿态估计** + **VLM 单图分类**，构建 14 类 CPR（心肺复苏）错误动作实时检测 Web 演示。
> 本文件为**唯一项目说明文档**。上游 ST-GCN 模型原始说明（`OLD_README.md` / `README.md`）已摘要于本文档；完整原文保留在本地归档目录 `_redundant_review_20260819/`（**不入库**，仅本机留存）。

---

## 1. 目录结构

```
CPROldshen/
├── README.md                      # 本文件（唯一交接文档）
├── .gitignore                     # Git 忽略规则：数据集/权重/推理产物/归档不入库（见 §9）
├── st-gcn-master/                 # 上游 ST-GCN + 本项目推理扩展（核心工作区）
│   ├── infer_demo/                # ★ 实时演示 + 工具脚本（主要改动都在这）
│   │   ├── app.py                 # Flask 入口（端口 5000，模型路径写死，见 §4）
│   │   ├── infer_service.py       # 实时推理：骨架缓冲/场景切换重置/KNN投票/Top-2显示
│   │   ├── infer.py               # 离线批量推理（输出 outputs/）
│   │   ├── vlm_service.py         # Ollama qwen3-vl 单图分类（9 类空间姿态）
│   │   ├── knn_match.py           # KNN 投票匹配
│   │   ├── static/  templates/    # Web UI（/live 实时检测页、/vlm 分类页）
│   │   ├── generate_demo_videos_from_mydata.py  # mydata 图片序列 → 1080p demo 视频
│   │   ├── combine_demo_videos.py               # 多段 demo 视频拼接
│   │   ├── convert_demovideos.py                # 4K mp4 → 1080p 转码
│   │   ├── outputs/               # 离线推理产物（1.8G，不入库，infer.py 可重生成）
│   │   └── uploads/               # 上传临时目录（运行时，Flask 需存在）
│   ├── work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt  # ★ 推理主权重（不入库，见 §9；UI 检测唯一加载，其余 epoch/ST_GCN0.001 已清理）
│   ├── mydata/                    # 合成视频源图片序列（10G，不入库，demo 已生成到 CPR-Coach-main）
│   ├── data/my_dataset/out/       # ★ label_map.json / class_names.json / train_label.pkl 等（KNN 必需，已入库）
│   ├── extract_pipeline/          # 数据抽取流水线（SupC14 整理等）；详细部署见「数据抽取与服务器部署.md」
│   └── ...（上游 ST-GCN 代码：feeder/ net/ models/ processor/ torchlight/ tools/ 等）
├── ultralytics-main/              # 上游 YOLO 源码（yolo26m-pose.pt 在 pose/，47M，权重不入库，见 §9）
└── Ollamaproject/                 # VLM 命令行分类（cpr_vlm_classify.py）
```

> `ultralytics-main/` 为第三方依赖源码，保留其自带 README 属正常（如同不删依赖库文档）。

---

## 2. 环境与依赖

- **Conda 环境**：`oldshen`（Windows，本地推理/演示主用；含 `ollama` Python 包）；`depth`（Linux，torch2.0.1+cu117，远程训练主用）。
- **Python 路径**（本地）：`C:\Users\11137\miniconda3\envs\oldshen\python.exe`
- **Ollama**：本地需运行 Ollama 并拉取 `qwen3-vl:2b`（端口 11434）。VLM 调用必须传 `num_ctx=16384`，否则报 `exceed_context_size`（默认 4096）。
- **ffmpeg**：用 `imageio-ffmpeg` 自带二进制（`envs/oldshen/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`）。
- 上游依赖见 `st-gcn-master/requirements.txt`。

---

## 3. 14 类动作（SupC00–SupC13）

类别中文名取自 `data/my_dataset/out/class_names.json`（与训练标签一致）：

| 编码 | 动作 | 检测难度（纯 2D 骨架） |
|---|---|---|
| SupC00 | 正确动作 | 🟢 高（基准姿态） |
| SupC01 | 双手重叠 | 🔴 低（手掌根/锁扣，骨架不可见） |
| SupC02 | 握拳按压 | 🔴 低（拳头/掌根点位重合） |
| SupC03 | 单手按压 | 🟢 高（单手悬空，一眼可辨） |
| SupC04 | 手臂弯曲 | 🟢 高（肘夹角直接） |
| SupC05 | 手臂倾斜 | 🟡 中（易与弯曲/正确混淆） |
| SupC06 | 跳跃按压 | 🟡 中（时间维度特征） |
| SupC07 | 蹲姿按压 | 🟢 高（身体高低几何） |
| SupC08 | 站立按压 | 🟢 高 |
| SupC09 | 位置偏移 | 🟡 中（手相对身体位置） |
| SupC10 | 按压不足 | 🔴 低（按压深度，2D 无 z 信息） |
| SupC11 | 频率过慢 | 🔴 低（节奏相对量） |
| SupC12 | 按压过度 | 🔴 低（按压深度） |
| SupC13 | 随机位置按压 | 🟡 中 |

**规律**：越接近"身体姿态几何"越准（蹲/站/单手/弯臂/正确）；越依赖"手部细节/节奏/按压深度"越弱（拳头/重叠/频率/深浅）。

---

## 4. 运行实时演示（Flask `/live`）

1. 启动：`cd st-gcn-master/infer_demo && python app.py`（后台运行，加载模型约 10–20s）。
2. 访问 `http://127.0.0.1:5000/live`。
3. 左侧"输入视频"卡片：点击或拖拽上传 mp4（H.264 Baseline 最佳）；或填本地路径。
4. 点视频自带 ▶ 播放，前端逐帧抽帧 → 后端推理 → 画施救者绿框 + Top-2 类别标签。

**`app.py` 写死的必需资产路径（勿删/勿移动）**：
- `stgcn_model`：`work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt`（`app.py:25`，推理主权重）
- `yolo_model`：`../ultralytics-main/pose/yolo26m-pose.pt`（`app.py:26`）
- `train_label`：`data/my_dataset/out/train_label.pkl`（`app.py:29`，KNN 训练数据）

> ⚠️ `app.py` 以 `debug=False` 启动会**缓存模板**，改了 `templates/` 或 `static/` 后必须**重启 Flask 进程**才生效。

---

## 5. 实时推理关键逻辑（`infer_service.py`）

- **只检施救者**：`select_cpr_person`（tools/yolo2stgcn.py）按 bbox 高宽比排除躺地患者，只画施救者骨架。
- **骨架缓冲**：`_MIN_BUFFER=300`、`_MAX_SEQ=300`、`_SLIDE_STEP=30`（`infer_service.py:438-442`）。累计 300 帧才进入 ready 正式推理；每 30 帧滑动更新一次。
- **延迟来源**：前端串行抽帧（`live.js` `setInterval(tick,100ms)` + `if(inflight)return` 背压），实际抽帧率 = min(10fps, 后端速度)。`300帧 ÷ ~15fps ≈ 20s` 首次出确定结果（缓冲期 ≥30 帧已有 provisional「估」）。
- **场景切换自动重置**：`_maybe_scene_reset` 检测硬切（关节平均位移 >0.16 且 ≥60% 关节同步大幅位移）→ 清空旧缓冲，解决拼接视频精度暴跌。带 24 帧去抖。
- **Top-2 显示**：ready 阶段 KNN（k=5 距离倒数加权）取 Top-2，`live.js drawSkeleton` 框上方画两行（Top1 绿 / Top2 灰）。
- **VLM 独立实验**：`vlm_service.py` 用 `qwen3-vl:2b` 对 9 类空间姿态单图分类（排除时序类 6/10/11/12/13），JSON 输出，Web `/vlm` + CLI `Ollamaproject/cpr_vlm_classify.py`。

---

## 6. 工具脚本

| 脚本 | 用途 |
|---|---|
| `generate_demo_videos_from_mydata.py` | 从 `mydata/SupCxx/r00_ch0/*.jpg` 合成 1080p H.264 Baseline/yuv420p(faststart) demo 视频（≥350 帧） |
| `combine_demo_videos.py <输出> <输入1..n>` | 多段 demo 拼接为一个（concat 滤波器重编码） |
| `convert_demovideos.py` | 4K mp4 → 1080p 转码 |
| `infer.py` | 离线批量推理，输出 `outputs/`（annotated.mp4 + probs/skel json） |

> ⚠️ ffmpeg 仅 `format=yuv420p` 仍会被标成全幅 `yuvj420p`，必须加 `setrange=limited` + `-color_range 1` 才是限幅 `yuv420p(tv)`，与参考视频一致。

---

## 7. 已知局限与后续建议

- **延迟**：300 帧缓冲导致首次/场景切换后约 20s 才出确定结果。可考虑降 `_MIN_BUFFER`（如 180）缩短等待（姿态类精度几乎不掉，节奏类略降）。
- **精度**：姿态类高、深度/节奏类低（单目 2D 无 z 信息）。建议引入深度相机或多视角，或把"按压深度"类改为时序/力信号判断。
- **运行资产绑定**：`app.py` 模型路径写死，交接后若移动文件需同步改 `app.py:25-29`。
- **大目录**：`mydata/`(10G)、`outputs/`(1.8G) 已移至归档或保留，按需清理（均可重生成）。
- **Ollama**：VLM 依赖本地 Ollama 服务，部署机器需预装并拉取 `qwen3-vl:2b`。

---

## 8. 交接状态

- 已验证：实时 `/live` 检测、VLM `/vlm` 分类、demo 视频生成/拼接、场景切换重置均可用。
- 清理：缓存/测试脚本/散落说明已移至 `_redundant_review_20260819/`；上游两个 README 摘要进本文档后移归档。
- 未动（按需）：`mydata/`、`outputs/`。
- 已清理：`work_dir` 冗余 `.pt` 权重及整个 `ST_GCN0.001` 训练产物（config.yaml/log.txt），**仅保留 UI 检测用的 `ST_GCN/epoch30_model.pt`**，其余 9 个 epoch 权重 + 目录移至归档。
- GitHub 整理（2026-08-19）：顶层 `.gitignore` 配置完成，`git init`（分支 `main`）+ 初始 commit 已提交；权重/数据集/推理产物不入库（见 §9/§10）。
- 数据抽取/服务器部署详细流程见 `st-gcn-master/extract_pipeline/数据抽取与服务器部署.md`（原 `README_server.md` 改名保留，不计入"唯一 README"）。

---

## 9. 模型权重获取（不入库，clone 后需自行放置）

两个运行必需的 `.pt` 权重均被 `.gitignore` 排除（规则 `*.pt`），**不随 GitHub 仓库分发**。clone 后需手动获取并放置到 `app.py` 写死的路径：

| 权重 | 大小 | 放置路径（相对仓库根） | 获取方式 |
|---|---|---|---|
| `yolo26m-pose.pt` | 47M | `ultralytics-main/pose/yolo26m-pose.pt` | ultralytics 官方权重：`pip install ultralytics` 后执行 `python -c "from ultralytics import YOLO; YOLO('yolo26m-pose.pt')"` 自动下载，或从 ultralytics 官方 Release 获取 |
| `epoch30_model.pt` | 12M | `st-gcn-master/work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt` | 本项目 ST-GCN **训练产物（非公开）**：由交接人线下提供；或按 `extract_pipeline/` + `config/st_gcn/` 流程自行重训 |

> KNN 数据（`data/my_dataset/out/*.npy/.pkl`，约 7M）**已入库**，clone 后无需额外操作即可跑 `/live` 的骨架/类别部分（缺权重时仅模型加载失败）。

---

## 10. GitHub 仓库说明

- 仓库已用顶层 `.gitignore` 排除：`mydata/`(10G)、`**/outputs/`(推理产物 1.8G)、`**/*.mp4`、`*.pt`(权重)、`_redundant_review_20260819/`(本地归档)、`__pycache__`/`*.py[cod]`、`.ultralytics_config/`。
- `ultralytics-main/` 与 `st-gcn-master/` 为 vendored 上游库（YOLO / ST-GCN），保留其自带 LICENSE / README；本项目改动集中在 `st-gcn-master/infer_demo/`、`extract_pipeline/`、`ultralytics-main/pose/`（`pose_estimate.py`、`ui/pose_ui.py`）。
- 提交策略：常规 `git add . && git commit` 即可，权重/数据不会进暂存；如后续想分发权重，建议走 GitHub Releases 或 Git LFS。
