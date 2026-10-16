# music-to-video

把一首歌做成一条跟着节拍走的地图：一颗球在每个拍点撞上预先放好的障碍，导出带原音频的 MP4。

三种实现（Python、浏览器、Rust）都跑通了同一条流水线，实测和取舍写在 [docs/tech-comparison.md](docs/tech-comparison.md)。正式工具用 **Python：librosa 检拍，cairo 离线绘制，ffmpeg 编码**。地图画面按 2 倍分辨率绘制再缩小，30 秒 1080p 成片大约 100 秒，节拍 F1 0.984。

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
music2video -i song.mp3 -o out.mp4 --aspect 16:9 --theme neon
music2video -i song.mp3 -o soft.mp4 --theme pastel
music2video -i song.mp3 -o story.mp4 --aspect 9:16 --theme auto --seed 7
python -m music2video -i assets/test-track.wav -o out.mp4 --aspect 16:9
music2video -i song.mp3 -o field.mp4 --scene balls
```

| 参数 | 含义 |
| --- | --- |
| `-i, --input` | wav、mp3，或 ffmpeg 能读的其它音频 |
| `-o, --output` | 输出的 mp4 |
| `--aspect` | `16:9` 为 1920×1080，`9:16` 为 1080×1920 |
| `--scene` | `map`（默认）一颗球走生成的地图；`balls` 是原来的多球场地；`city` 夜城生长，`piano` 音符瀑布，`stars` 星座 |
| `--theme` | 整支片子只用一种皮肤。`neon` 霓虹、`arcade` 街机、`piano` 琴键、`shatter` 冲击、`pastel` 柔彩。`auto` 按整首歌的能量选一种，不会中途混用 |
| `--fps` | 默认 30 |
| `--seed` | 地图走向和障碍种类的随机种子，默认 42 |
| `--max-seconds` | 只处理歌曲开头的若干秒 |
| `--beat-mode` | `default` 用 librosa 的节拍；`soft` 在更快的一倍速上，若半速网格的 onset 不更差，就改用半速。鼓点清楚的歌不会被改慢 |
| `--crf` | x264 质量，默认 20 |
| `--title`, `--artist` | 可选的角落标题，默认不显示，开头淡入再淡出 |
| `--dump-beats` | 把节拍、段落和每个障碍写成 JSON |

默认画面是一张随歌曲长出来的地图，整支片子只用一个主题。路径仍按拍点摆好，球心在拍上正好贴住表面。主题改的是撞上去的东西、以及这一下的手感，不是把所有障碍掺在一起。

| 主题 | 飞行物 | 手感 | 障碍 | 撞击 |
| --- | --- | --- | --- | --- |
| `neon` 霓虹 | 霓虹箭头 | 轻、回弹快，几乎不变形 | 只有墙 | 冲击波环 |
| `arcade` 街机 | 金属弹珠 | 挤压利落，马上弹开 | 圆桩、弹垫、转杆 | 火花 |
| `piano` 琴键 | 音符 | 先拉长再收回 | 钢琴键、平台 | 柔光 |
| `shatter` 冲击 | 会翻滚的橡胶方块 | 重，撞上停一停再慢慢复原 | 碎块、墙 | 碎片和屏幕震动 |
| `pastel` 柔彩 | 肥皂泡 | 慢、飘，泡始终是圆的，撞上时轻轻胀一下 | 墙、平台 | 柔光 |

手感写在主题的 `Feel` 里：质量、挤压幅度、拉伸、弹簧劲度、阻尼、停顿、拖尾。这些只改变变形，不改拍点。柔和的歌可以用 `--theme pastel`，或者 `--theme auto` 让能量偏低的歌自己落到柔彩或琴键。

同一首片子不会换皮肤，但主题内部会变。平静段更小、更淡、飞得更远；激烈段更大、更亮、障碍排得更密。变体只来自本主题：霓虹是长短墙、转角和双墙，柔彩是不同长度和弧度的平台，琴键是单音、和弦和黑白键，冲击是裂纹、厚块和叠块。每四个强拍有一次仍属于这个主题的花样（粗冲击波、气泡裂开再合上、更大的碎裂、琴键琶音）。路径在回廊、轻微弧线和偶尔的长直道之间切换，拍点仍然落在表面上。

撞过的障碍留在地图上，并且仍是实体。后面的航段会绕开已经放下的墙和平台；飞过的圆盘只在当前这一拍贴住接触面，不会再穿进旧障碍。新的落点会离开已经挤在一起的障碍，两条障碍之间留出明显空隙；这一边还没走到空地，就不会拐回旧墙堆里，路本身仍会转弯，不会一直朝一个方向飘。画面里障碍多起来时，更早的那些会淡掉。镜头阻尼跟着飞行物，并按能量轻轻缩放。

画面按主题用一套配色：圆角障碍、内渐变和细高光、飞行物的光晕和收窄拖尾、轻微暗角和颗粒。角标默认关闭，需要时加 `--title` 和 `--artist`。

`--scene balls` 保留原来的多球场地：每拍反向并换色，强拍爆粒子。

另外三种画面不走球和地图，节拍或旋律仍然对齐，并且用同一套成片处理：2 倍分辨率绘制再缩小、柔光、暗角、颗粒，镜头只做阻尼跟随。

| `--scene` | 画面 | 对齐 |
| --- | --- | --- |
| `city` | 夜色里一座城市长出来。每一拍立起一栋楼或点亮窗户，低音铺亮整条街，楼有多高跟着这一段的能量。结尾镜头已经拉开，天际线站在画面里 | 楼在拍点上长到预定高度 |
| `piano` | 像瀑布琴一样，音符落到键盘上，键亮起，长短等于这个音持续了多久 | librosa 的音高、色度和 onset。音头在 onset 上碰到键 |
| `stars` | 一个音点亮一颗星，一句连成一个星座，夜空慢慢漂，星星轻轻闪 | 星星在音符到来时达到最亮 |

这三支片子不跟 `--theme` 换肤，各用自己的夜色。地图和多球场不受影响。

加一种障碍或撞击效果：在 `music2video/registry.py` 里调用 `register_object` / `register_effect`。加一种皮肤：在 `music2video/themes.py` 里 `register_theme`，写上飞行物、`Feel`、障碍和效果。分析、场景、绘制、编码互不调用。

合成测试曲（126 BPM，节拍时间已知）：

```bash
python tools/make_test_track.py
python -m unittest tests.test_pipeline
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `music2video/` | 正式 CLI。`analysis.py`、`notes.py`、`mapscene.py`、`forms.py`、`themes.py`、`registry.py`、`scene.py`、`render.py`、`encode.py`、`cli.py` |
| `prototypes/python` | librosa + cairo 原型 |
| `prototypes/web` | Web Audio AnalyserNode + Canvas + WebCodecs |
| `prototypes/rust` | rustfft + tiny-skia + ffmpeg |
| `legacy/` | 原来的 pygame 演示，只能开窗口，不能导出视频 |
| `docs/tech-comparison.md` | 实测对比和推荐理由 |
| `tools/make_test_track.py` | 生成 30 秒测试音频和标准节拍 |

原型复现步骤见对比文档。仓库里的 mp3 是旧演示用的素材，评分和示例视频用的是脚本生成的测试曲。
