# music-to-video

把一首歌做成一条跟着节拍走的地图：一颗球在每个拍点撞上预先放好的障碍，导出带原音频的 MP4。

三种实现（Python、浏览器、Rust）都跑通了同一条流水线，实测和取舍写在 [docs/tech-comparison.md](docs/tech-comparison.md)。正式工具用 **Python：librosa 检拍，cairo 离线绘制，ffmpeg 编码**。30 秒 1080p 成片大约 7 秒，节拍 F1 0.984。

## 安装

需要 Python 3.10+、ffmpeg，以及系统里的 cairo。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Debian/Ubuntu 额外需要：
# sudo apt install ffmpeg libcairo2 pkg-config python3-dev
```

## 用法

```bash
music2video -i song.mp3 -o out.mp4 --aspect 16:9
music2video -i song.mp3 -o story.mp4 --aspect 9:16 --seed 7
python -m music2video -i assets/test-track.wav -o out.mp4 --aspect 16:9
music2video -i song.mp3 -o field.mp4 --scene balls
```

| 参数 | 含义 |
| --- | --- |
| `-i, --input` | wav、mp3，或 ffmpeg 能读的其它音频 |
| `-o, --output` | 输出的 mp4 |
| `--aspect` | `16:9` 为 1920×1080，`9:16` 为 1080×1920 |
| `--scene` | `map`（默认）一颗球走生成的地图；`balls` 是原来的多球场地 |
| `--fps` | 默认 30 |
| `--seed` | 地图走向和障碍种类的随机种子，默认 42 |
| `--max-seconds` | 只处理歌曲开头的若干秒 |
| `--crf` | x264 质量，默认 20 |
| `--dump-beats` | 把节拍、段落和每个障碍写成 JSON |

默认画面是一张随歌曲长出来的地图：

- 球以恒定速度走折线。每一拍的落点上放一个障碍，法线按入射方向和离开方向算好，所以球心在拍点上正好贴住表面。含拍的那一帧按拍点时刻来画，命中误差是 0。
- 障碍按段落挑选。分析把歌曲切成 2 秒一段，用 RMS 标成 calm / mid / intense。平静段多用墙、平台、圆桩和钢琴键；激烈段多用弹垫、碎块、转杆和圆桩。强拍用更大的障碍。
- 撞上之后障碍留在地图上并发光。镜头跟着球走，后面的路是事后才留下来的。
- 撞击效果放在注册表里：冲击波、火花、碎块、光晕、色差闪、球的挤压拉伸、颜色和形状变化。强拍再加屏幕震动。平静段的效果更轻。
- 背景跟着能量走：视差星点和格子、径向脉冲，以及画面底部的频谱条。配色随段落切换。

`--scene balls` 保留原来的多球场地：每拍反向并换色，强拍爆粒子。

加一种障碍或撞击效果：在 `music2video/registry.py` 里调用 `register_object` / `register_effect`。对象声明自己出现在哪些段落、命中时触发哪些效果，以及一个 draw 函数。分析、场景、绘制、编码互不调用。

合成测试曲（126 BPM，节拍时间已知）：

```bash
python tools/make_test_track.py
python -m unittest tests.test_pipeline
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `music2video/` | 正式 CLI。`analysis.py`、`mapscene.py`、`registry.py`、`scene.py`、`render.py`、`encode.py`、`cli.py` |
| `prototypes/python` | librosa + cairo 原型 |
| `prototypes/web` | Web Audio AnalyserNode + Canvas + WebCodecs |
| `prototypes/rust` | rustfft + tiny-skia + ffmpeg |
| `legacy/` | 原来的 pygame 演示，只能开窗口，不能导出视频 |
| `docs/tech-comparison.md` | 实测对比和推荐理由 |
| `tools/make_test_track.py` | 生成 30 秒测试音频和标准节拍 |

原型复现步骤见对比文档。仓库里的 mp3 是旧演示用的素材，评分和示例视频用的是脚本生成的测试曲。
