# 浏览器里的 music-to-video

歌不上传，片子不在服务器上画。用户在自己的浏览器里完成四件事：分析、选模板、实时预览、导出 mp4。服务器以后只做模板市场和给用户自己的 AI 代理一个很轻的接口，渲染成本保持接近于零。

`web/` 是这个约定的第一版原型。Python 命令行（`music2video/`）继续负责现在这几支片子，两边通过同一套时间契约对齐，而不是共用绘制代码。

## MusicData

分析只做一次，结果是一份可序列化的 JSON。模板只读这份数据，不再自己打开音频。这样预览、导出、以后的市场模板和本地 CLI 看到的是同一首歌。

```ts
interface MusicData {
  version: 1;
  duration: number;          // 秒
  sampleRate: number;        // 分析用的采样率，不是播放文件的采样率
  tempo: number;             // BPM
  meter: number;             // v1 固定 4，一小节四拍
  beats: number[];           // 每一拍的时间
  downbeats: number[];       // 每小节第一拍。相邻两个 downbeat 之间就是一个 bar
  onsets: { time: number; strength: number }[];  // strength 在 0..1
  loudness: { times: number[]; values: number[] }; // 约 1.2 秒平滑后的响度，0..1
  bands: {
    times: number[];
    low: number[];           // 20–250 Hz
    mid: number[];           // 250–2000 Hz
    high: number[];          // 2000–8000 Hz
  };
  sections: {
    start: number;
    end: number;
    energy: number;
    label: "intro" | "verse" | "chorus" | "bridge" | "outro";
  }[];
  extensions?: {
    analyzer?: string;
    strongBeats?: number[];  // beats 里偏重的那些拍，可选
  };
}
```

约定：

- `version` 改了，表示旧模板不能直接读。加在 `extensions` 里的字段不算改版本，模板必须忽略自己不认识的键。
- 响度是包络，不是每一个音头。音头的瞬态在 `onsets` 里。频段比响度更灵敏，大约 0.35 秒的平滑，用来让画面跟着低音或高频呼吸。
- 时间全部是秒，从文件开头算。模板不得假设均匀采样，要用插值。

### 分析器

原型用的是浏览器里现成、能跑完一整首歌的一条管线，而不是 Essentia.js。

1. `decodeAudioData` 解码。
2. `OfflineAudioContext` 混成单声道并重采样到 22050 Hz。
3. 1024 点 FFT、512 点跳跃，算频谱通量、RMS，以及低/中/高频能量。
4. 节拍用 Ellis 的动态规划（仓库里早期 web 原型已经用过同一套跟踪器）。重拍看拍点上的高频通量。Downbeat 在 4 拍相位里选低音能量更高的那一相。
5. 段落用平滑响度切成 intro / verse / chorus / bridge / outro。歌如果几乎一样响，就把中间一段标成副歌，避免整首没有结构。

Essentia.js 或编译成 WASM 的 aubio 可以以后换掉这一步，尤其是要加音高和色度的时候。换分析器不许改模板函数的签名，只许改 `MusicData` 的内容和 `extensions`。钢琴瀑布需要音高，那是下一步要加进 `extensions` 的可选音轨（`pitches`、`chroma`），不是 v1 的必填字段。

## 模板 API

这是平台和模板之间的全部接口。

```ts
defineTemplate({
  meta: {
    id: string,
    name: string,
    author: string,
    version: string,
    description: string,
  },
  tier: "2d" | "webgpu",
  params: {
    [name]: NumberParam | ColorParam | EnumParam | BooleanParam,
  },
  setup(ctx, music, params) {
    return state;          // 只跑一次。雨滴落点、镜头曲线都在这里算好
  },
  render(ctx, t, state, music, params) {
    // 画 t 这一帧。必须是纯函数。
  },
});
```

`params` 的每一项：

| type | 字段 | 面板控件 |
| --- | --- | --- |
| `number` | `label`, `min`, `max`, `step`, `default` | 滑杆 |
| `color` | `label`, `default`（`#rrggbb`） | 取色 |
| `enum` | `label`, `options: { value, label }[]`, `default` | 下拉 |
| `boolean` | `label`, `default` | 开关 |

`scope` 是 `"plan"` 或 `"frame"`，默认 `"plan"`。`"plan"` 会重新跑 `setup`（落点、密度、种子）。`"frame"` 只重画当前帧（月亮位置、水色），拖滑杆时不能卡。

### 纯函数

同一个 `t` 必须画出同一帧。`render` 不能读上一帧留下的粒子、不能累加镜头弹簧、不能用 `Math.random()` 或 `Date.now()`。

需要“阻尼”的东西在 `setup` 里沿时间积出来，存成曲线，`render` 只插值。雨夜的镜头就是这样做的：60 Hz 上把弹簧积完，预览拖到任意一秒都和从头播到那里一样。

随机数只用平台给的 `ctx.rng()`，范围是 `[0, 1)`。`setup` 里它顺着一个种子往下走。`render` 里平台按 `(模板 id, 种子, floor(t * 30))` 重新播种，所以某一帧用了随机数也不会被前一帧影响。雨夜把随机都放在 `setup`，画面里的颗粒也由这个时间种子决定。

### 两层上下文

`ctx.tier === "2d"`：

- `ctx` 是 `CanvasRenderingContext2D`
- `width` / `height` 是这一帧的像素大小
- `post` 是 `{ glow, grain, vignette }`。模板在 `render` 末尾写下这一帧要多亮的柔光。平台在模板返回之后做阈值柔光、暗角和颗粒，和 Python 成片那一步同一套，模板自己不必再做

`ctx.tier === "webgpu"`：

- `device`、`context`、`format`
- 同样有 `rng` 和 `post`
- 浏览器没有 WebGPU 时，平台把 `tier` 降成 `"2d"`。模板若只实现了 WebGPU，要在 `setup` 里明确失败；平台侧的降级是“能画 2D 就不把页面弄黑”

雨夜走 2D。城市、星空以后如果要大量光斑，可以走 WebGPU，并用 2D 画一个简化版当后备。

## 预览

页面解码出 `AudioBuffer` 之后，用一个 `<audio>` 播放同一份文件。每一帧读 `audio.currentTime`，调用 `render(ctx, t, ...)`。拖动进度条时立刻用滑杆上的时间画一帧，不等 `seeked`。因为 `render` 不依赖上一帧，拖到副歌中间不会出现“粒子还停在前奏”的情况。

暂停时停在当前 `t`。示例数据（还没上传歌）用一个 24 秒的合成节奏让水面自己动，上传后整段换成那首歌的 `MusicData`。

## 导出

导出不用服务器。

1. 按导出分辨率重新 `setup` 一次（落点跟着画布宽度缩放，和预览是同一构图）。
2. 从 `t = 0` 起，每 `1/30` 秒调用同一次 `render`，再跑同一套 `post`。
3. `VideoFrame` 送进 `VideoEncoder`（H.264）。
4. 音频用 `OfflineAudioContext` 转到 48 kHz 立体声，`AudioEncoder` 编 AAC。
5. `mp4-muxer` 把两条轨收成一个 mp4，`fastStart: "in-memory"`，浏览器直接下载。

原型导出 1280×720、30 fps。浏览器没有 WebCodecs 或 H.264 时，按钮会说明原因，预览仍然可用。

## 沙箱

用户写的模板和官方模板不是同一份信任。官方模板（现在的雨夜）跑在页面里。市场上载的模板要进隔离环境：

- 一个 `sandbox` 且不带 `allow-same-origin` 的 iframe，或一个没有网络权限的 Worker + `OffscreenCanvas`。
- CSP：`default-src 'none'`，不许 `fetch`、不许读 cookie 和用户文件。音频和 `MusicData` 由宿主用 `postMessage` 送进去。
- 模板代码只能看到上面的 `defineTemplate` 上下文，看不到 `window` 上的文件、账号和导出按钮。
- 宿主每帧把 `t` 送进去，拿回 `ImageBitmap`。超时或抛错就停掉这一帧并标成模板错误，不拖垮页面。

`render` 是纯函数，所以从主线程挪进 Worker 不用改模板。原型还没把雨夜放进 Worker，接口已经按这个边界写了。

## 模板市场（稍后）

部署在 Cloudflare 上，不承担渲染。

| 部分 | 用途 |
| --- | --- |
| Workers | HTTP API：列表、详情、上传、点赞、举报 |
| R2 | 模板包（一个 JS 模块）和预览短片 |
| D1 | 账号、模板、版本、点赞、下载计数、举报 |

表大致是：`users`、`templates`、`template_versions`（版本、R2 key、校验状态）、`likes`、`download_counts`、`reports`。

上传流程：

1. 作者提交包和说明。
2. Worker 做静态检查：有 `defineTemplate`、`render` 不引用禁止的全局、体积上限。
3. 用固定的 8 拍夹具在无头浏览器里跑几帧（还是作者浏览器或 CI，不是给每个访客渲染）。失败的版本不公开。
4. 通过后写成一条 `template_versions`，预览片放进 R2。
5. 任何人可以举报。举报进 `reports`，下架是人来做的。

访客只下载模板包，在自己的浏览器里跑。

## 用户自己的代理（稍后）

远程 MCP 放在 Workers 上，工具只有这些，没有“帮我渲染”：

| 工具 | 作用 |
| --- | --- |
| `get_template_spec` | 返回这份文档里的类型和一条最小例子 |
| `list_examples` | 官方模板的源码（雨夜是第一条） |
| `validate_template` | 静态检查，和市场上传用的是同一套 |
| `publish_template` | 返回一个受限的上传地址 |

看画面在作者自己的机器上：

```bash
npx m2v preview template.ts song.mp3 --at 12.5 --out frame.png
npx m2v preview template.ts song.mp3 --clip 8 --out clip.mp4
```

CLI 打开本地无头浏览器，跑同一个 `render`。服务器不碰音频，也不出帧。

## 现有 Python 场景怎么迁

| Python | 模板 | 层 | 用到的 MusicData | 备注 |
| --- | --- | --- | --- | --- |
| `rain` | `rain`（已在 `web/`） | 2D | 拍、重拍、音头、响度、段落 | 落点、风和镜头都在 `setup` |
| `city` | `city` | 2D，楼多了可改 WebGPU | 拍、重拍、响度、频段 | 楼的生长时刻在 `setup` |
| `stars` | `stars` | 2D 或 WebGPU | 音头、段落 | 星座折线在 `setup` 里保证不相交 |
| `piano` | `piano` | 2D | 拍、音头，外加以后的 `extensions.pitches` / `chroma` | v1 没有音高，先不迁 |
| `map` / `balls` | `map` | 2D | 拍、重拍、种子 | 地图在 `setup` 生成 |

Python 里那些“跟上一帧弹簧”的镜头，迁的时候都改成 `setup` 里积好的曲线，否则网页预览没法随意拖动。

## 路线

1. **现在。** 浏览器分析、`MusicData`、模板 API、参数面板、可拖动的预览、雨夜、本地 mp4。见 `web/`。
2. **CLI。** `m2v preview`，和网页用同一份渲染，方便人和代理看关键帧。
3. **把 city、stars、map 迁进来。** 钢琴等音高扩展。
4. **沙箱。** 用户模板进 Worker / iframe。
5. **市场。** Workers + R2 + D1，上传校验和举报。
6. **MCP。** 只暴露读规范、看例子、校验、发布。

## 跑原型

```bash
cd web
npm install
npm run dev
```

打开终端里打印的本地地址。上传一首歌，等分析结束，拖时间轴，右侧滑杆会改雨、风和水色。`导出 MP4` 在本地编码。`下载 MusicData` 把分析 JSON 存下来。

`npm run build` 之后可以用 `npm run preview` 看打包结果。无头截图：

```bash
node scripts/capture.mjs /path/to/song.mp3 /tmp/rain-frames
```

还没做的：模板不在 Worker 里跑，导出分辨率固定 1280×720，段落是响度启发式而不是真正的曲式分析，钢琴需要的音高还没进 `MusicData`。
