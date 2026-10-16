# music-to-video

把一首歌做成跟着节拍跳动的小球动画，导出带原音频的 MP4。

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
music2video -i song.mp3 -o story.mp4 --aspect 9:16
python -m music2video -i assets/test-track.wav -o out.mp4 --aspect 16:9
```

| 参数 | 含义 |
| --- | --- |
| `-i, --input` | wav、mp3，或 ffmpeg 能读的其它音频 |
| `-o, --output` | 输出的 mp4 |
| `--aspect` | `16:9` 为 1920×1080，`9:16` 为 1080×1920 |
| `--fps` | 默认 30 |
| `--seed` | 小球初始位置，默认 42 |
| `--max-seconds` | 只处理歌曲开头的若干秒 |
| `--crf` | x264 质量，默认 20 |
| `--dump-beats` | 把检出的节拍写成 JSON |

画面：

- 每个节拍上，小球速度反向并略微加快，相当于跟着拍子弹一下
- 节拍时换色，并闪一圈白边
- 强拍（高频 onset 不低于全部节拍中位数的 1.35 倍）从每个球爆出一簇粒子
- 小球碰到画面边缘也会反弹

再加一种反应：在 `music2video/scene.py` 里写一个带 `on_beat(self, scene, strong)` 的类，放进 `Scene.create` 的 `effects` 列表。分析、绘制、编码互不调用。

合成测试曲（126 BPM，节拍时间已知）：

```bash
python tools/make_test_track.py
python -m unittest tests.test_pipeline
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `music2video/` | 正式 CLI。`analysis.py`、`scene.py`、`render.py`、`encode.py`、`cli.py` |
| `prototypes/python` | librosa + cairo 原型 |
| `prototypes/web` | Web Audio AnalyserNode + Canvas + WebCodecs |
| `prototypes/rust` | rustfft + tiny-skia + ffmpeg |
| `legacy/` | 原来的 pygame 演示，只能开窗口，不能导出视频 |
| `docs/tech-comparison.md` | 实测对比和推荐理由 |
| `tools/make_test_track.py` | 生成 30 秒测试音频和标准节拍 |

原型复现步骤见对比文档。仓库里的 mp3 是旧演示用的素材，评分和示例视频用的是脚本生成的测试曲。
