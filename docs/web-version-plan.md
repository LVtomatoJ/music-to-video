# 网站版：歌在浏览器里变成片子

人上传一首歌，得到一条视频。也可以上传自己的模板，或者让自己的 AI 代理写模板。服务器不画帧。画在用户的浏览器里，或者在代理自己的机器上。网站放在 Cloudflare。

命令行（`music2video/`）继续能出片。网页不把 Python 搬上去跑，而是沿用 `web/` 里已经在用的同一套时间数据。

## 浏览器里做的事

上传的文件留在本机。`web/src/main.ts` 已经这样做了：`<input>` 读文件，`decodeAudioData` 解码，分析完把 `MusicData` 交给模板，`<audio>` 播同一份缓冲，导出用 WebCodecs。

| 现在的 Python | 到浏览器里 | 用什么 |
| --- | --- | --- |
| `analysis.py`：ffmpeg 解码，librosa 节拍、重拍、响度、频段、2 秒一段的能量 | 已有 `web/src/music.ts`。解码用 Web Audio，频谱用 `web/src/fft.ts`，节拍用 Ellis 动态规划（`web/src/beats.ts`） | 继续用这套。它够把预览跑起来 |
| `notes.py`：pyin 音高，薄的时候用色度补 | `MusicData.extensions.tones` 里已经有一条粗音高（频谱峰 + 色度），不是 pyin | 音高质量不够时，再把跟踪器编成 WASM 换掉这一步。只换 `extensions`，不改模板函数的签名 |
| `forms.py` / `rain.py` / `mapscene.py` 的布局（楼、星座、落点、路径） | 模板的 `setup`。星座这种“跟拍呼吸”的量在 `setup` 里沿时间积成曲线，`render` 只查表 | 和 `web/src/templates/stars.ts` 里镜头已经在做的一样。Python 里 `update` 逐帧阻尼的镜头不能直接搬，否则拖动进度条会对不上 |
| `render.py` 的 cairo 绘制和柔光 | 模板的 `render`。雨、城市、钢琴、星空、地图现在走 Canvas 2D。星点很多时可以另写 `tier: "webgpu"`，没有 WebGPU 就退回 2D | 柔光、暗角、颗粒留在 `web/src/post.ts`，模板只填 `post.glow` 这类数 |
| `encode.py` 的 ffmpeg | `web/src/export.ts`：`VideoEncoder`（H.264）+ `AudioEncoder`（AAC）+ `mp4-muxer`，浏览器下载 | 不做服务器编码。没有 WebCodecs 时按钮说明原因，预览仍然能看 |

`render(t)` 必须是纯函数：同一个 `t` 画出同一帧。不用 `Math.random()`、不用上一帧留下的粒子。随机只用平台给的 `ctx.rng()`。这样拖到副歌中间不会出现“粒子还停在前奏”。

## Cloudflare 上留的事

只做登记和分发，不碰音频，不出帧。

| 部分 | 做什么 |
| --- | --- |
| Pages | 托管 `web/` 的构建结果。这是第一步就要上去的 |
| Workers | 很薄的 HTTP：模板列表、详情、上传登记、登录。以后给代理一个同样薄的接口 |
| R2 | 模板包（一个 JS 模块）。作者若愿意，附上自己机器渲的几张预览图。文件是作者传上来的，不是我们渲的 |
| D1 | 用户、模板、版本、是否公开 |

不做的事：不存用户的歌，不跑分析，不为每个访客出预览视频，不为代理开一台渲染机。

登录用 Cloudflare Access 或 Workers 上的会话即可，先不要自己做一套账号系统。

## 模板是什么

一份模板是一个 JS 模块，形状就是 `web/src/template.ts` 里的 `defineTemplate`：

- `meta`：id、名字、作者、版本、一句话
- `tier`：`"2d"` 或 `"webgpu"`
- `params`：数字、颜色、枚举、开关。`scope: "plan"` 会重跑 `setup`（星座怎么摆），`scope: "frame"` 只重画当前帧（月色、光晕）
- `setup(ctx, music, params)`：跑一次，返回状态
- `render(ctx, t, state, music, params)`：画 `t` 这一帧

模板只读 `MusicData`（`web/src/music.ts`）。不认识的 `extensions` 字段必须忽略。`version` 变了才表示旧模板不能读。

官方模板（现在的雨、城市、钢琴、星空、地图）和用户模板不是同一份信任。

用户模板进沙箱：

- 没有网络的 Worker，加上 `OffscreenCanvas`。或者一个 `sandbox` 且不带 `allow-same-origin` 的 iframe
- CSP：`default-src 'none'`。不能 `fetch`，不能读 cookie 和本地文件
- 宿主用 `postMessage` 送进 `MusicData` 和当前的 `t`，拿回一帧 `ImageBitmap`
- 模板只能看见 `defineTemplate` 的上下文，看不见页面上的账号和导出按钮
- 一帧超时或抛错，丢掉这一帧并标成模板错误，页面继续

`render` 是纯函数，所以从主线程挪进 Worker 不用改模板。

上传时 Worker 做静态检查：有 `defineTemplate`、不引用 `fetch` / `document.cookie` / `XMLHttpRequest`、体积有上限。用一首固定的短夹具在作者浏览器里跑几帧，失败就不公开。这一步也不是我们的服务器在画。

## 外面的 AI 代理怎么接

代理在它自己的机器上写模板、出预览、看帧，改完再提交。服务器没有“帮我渲一帧”这个工具。

本地命令（和网页用同一份 `render`）：

```bash
npx m2v preview template.ts song.mp3 --at 12.5 --out frame.png
npx m2v preview template.ts song.mp3 --clip 8 --out clip.mp4
```

它打开本机的无头浏览器，跑同一个模板。歌和帧都不离开这台机器。

给代理的接口保持很少，风格像 MCP，放在 Workers 上，只有这些：

| 工具 | 作用 |
| --- | --- |
| `get_template_spec` | 返回 `defineTemplate` 的类型和一份最小例子 |
| `list_examples` | 官方模板源码。雨是第一份，星空跟着 |
| `validate_template` | 和上传用的是同一套静态检查 |
| `publish_template` | 返回一个限时的 R2 上传地址。代理把包传上去，不把歌传上来 |

代理的循环是：写模块 → `preview` 出几帧 → 自己看安静段、拍点、高潮 → 改 → `validate` → `publish`。看画面这一步不经过我们。

## 路线

1. **第一步，就做这两件。** 把现有 `web/` 构建结果放到 Cloudflare Pages，仍然不接 D1、不接登录。上传、分析、预览、导出都在浏览器里，和现在本地 `npm run dev` 一样。同时把 Python 星空这次的节拍反应补进 `web/src/templates/stars.ts`（亮起、连线、拍上的光沿星座走、星云随段落起伏），让网页和命令行是同一支片子。用已有的 `node scripts/capture.mjs` 对一首本地歌出几帧核对。
2. **本地预览命令。** `m2v preview`，给人和代理看指定时间的帧。仍然不上传。
3. **沙箱。** 用户模板进 Worker。官方模板先继续跑在页面里。
4. **登记。** Workers + R2 + D1：登录、模板列表、上传和静态检查。
5. **代理接口。** 只开上面那四个工具。

先不要做的：应用商店、账号等级、服务器端分析、为每个访客生成预览片、在网页里再做一套多球场景。
