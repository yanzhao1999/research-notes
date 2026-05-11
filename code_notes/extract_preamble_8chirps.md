# LoRa 8-Upchirp Preamble Extraction 笔记
## 1. 实验目标
本实验的目标是：从原始 IQ 文件中自动提取干净、对齐好的 LoRa preamble 片段。
每一个样本包含：
```text
8 个 LoRa upchirp

在当前参数下：

1 个 upchirp = 512 个 complex IQ samples
8 个 upchirp = 4096 个 complex IQ samples

这些提取出来的 8-upchirp 片段后续会被转换成 spectrogram，用作 spectrogram-based CNN 的输入，用于 LoRa RFFI 设备识别实验。

整体流程：

Raw IQ
  ↓
生成本地 upchirp / downchirp
  ↓
滑动窗口 dechirp + FFT
  ↓
用 peak_ratio_db 检测可能的 chirp 位置
  ↓
根据 packet 间隔去重
  ↓
对每个 coarse_start 做 local search
  ↓
找到 8 个 peak bin 完全一致的 preamble 起点
  ↓
截取 4096 点并保存

⸻

2. LoRa 参数

当前实验参数：

SF = 7
BW = 125e3
OS = 4
FS = BW * OS

对应含义：

SF = 7
BW = 125 kHz
OS = 4
FS = 500 kS/s

计算：

M = 2 ** SF
NSYM = int(M * OS)
N_PREAMBLE = 8
PREAMBLE_SAMPLES = N_PREAMBLE * NSYM

结果：

M = 128
NSYM = 512
PREAMBLE_SAMPLES = 4096

理解：

M 是 LoRa symbol 的理论 bin 数
NSYM 是一个 chirp 的实际采样点数
PREAMBLE_SAMPLES 是 8 个 chirp 的总采样点数

所以：

一个 chirp = 512 samples
一个 8-upchirp preamble = 4096 samples

⸻

3. 生成本地 upchirp 和 downchirp

生成 upchirp 的公式：

phase = 2 * np.pi * ((-bw / 2) * t + 0.5 * k * t**2)
upchirp = np.exp(1j * phase).astype(np.complex64)

其中：

t 是每个采样点对应的真实时间
T 是一个 chirp 的持续时间
k = bw / T 是扫频斜率

这个公式生成的是一个从：

-BW/2 到 +BW/2

扫频的 baseband upchirp。

生成 downchirp：

downchirp = np.conj(upchirp)

downchirp 的作用是用于 dechirp。

⸻

4. Dechirp + FFT 的核心原理

对于一个长度为 NSYM 的窗口：

seg = x[start:start + NSYM]

先做 dechirp：

y = seg * downchirp

如果 seg 中包含 LoRa upchirp，那么：

received upchirp × local downchirp

会抵消扫频变化，使原本的 chirp 变成近似单频信号。

然后做 FFT：

Y = np.fft.fft(y)
P = np.abs(Y) ** 2

其中：

Y 是复数频谱
P 是功率谱

如果 dechirp 后接近单频，那么功率谱 P 中会出现一个明显峰值。

计算：

peak_bin = np.argmax(P)
peak_power = np.max(P)
mean_power = np.mean(P)
peak_ratio_db = 10 * np.log10(peak_power / mean_power)

含义：

peak_bin：FFT 最大峰值所在的频率格子
peak_power：最大峰值功率
mean_power：平均频域功率
peak_ratio_db：峰值比平均功率高多少 dB

判断：

peak_ratio_db 高 → 这个窗口很可能包含 LoRa upchirp
peak_ratio_db 低 → 更可能是噪声或非 chirp 区域

⸻

5. FFT bin 的理解

FFT 后会得到很多频率格子，这些格子叫做 bin。

在当前实验中：

FS = 500000 Hz
FFT 长度 = 512

所以：

1 bin = FS / 512 ≈ 976.56 Hz

例如：

peak_bin = 507

表示 FFT 结果中第 507 个频率格子的能量最大。

注意：

bin 不是采样点编号
bin 是频率格子编号

⸻

6. 为什么要检查 8 个 chirp 的 peak bin？

LoRa preamble 由重复 upchirp 组成。

如果一个 8-upchirp 片段对齐正确，那么每个 chirp 经过：

dechirp → FFT

之后，peak bin 应该基本一致。

理想情况：

[507 507 507 507 507 507 507 507]

不好的情况：

[508 500 477 504 454 470 53 491]

所以：

8 个 peak_bin 越一致，说明 preamble 越对齐

这是判断提取质量的核心标准。

⸻

7. 滑动窗口检测

设置：

HOP = 32

含义：

滑动窗口每次移动 32 个 IQ samples

因为：

NSYM = 512
512 / 32 = 16

所以一个 chirp 长度内会检查 16 次。

滑窗过程：

x[0:512]
x[32:544]
x[64:576]
x[96:608]
...

每个窗口都计算一次 peak_ratio_db。

保存三个数组：

metrics：每个窗口的 peak_ratio_db
starts：每个窗口的起始 sample index
peak_bins：每个窗口的 FFT peak bin

⸻

8. strong windows 和去重

设置阈值：

THR_DB = 20

找出强窗口：

high_idx = np.where(metrics > THR_DB)[0]

这些窗口说明：

这里附近可能有 LoRa chirp

但是同一个 packet 附近会产生很多连续的 high windows，所以需要去重。

因为实验中 packet 大约每 1 秒发送一次，所以设置：

MIN_PACKET_GAP_S = 0.8

只保留相隔至少 0.8 秒的 coarse candidate。

这样可以得到：

candidate_coarse_starts

它们是每个 packet 附近的粗略起点。

⸻

9. Coarse start 和真正 preamble 起点的区别

滑窗检测得到的是：

coarse_start

它只说明：

这里附近有一个强 chirp

但是它不一定是：

8 个 upchirp 的第一个起点

例如之前检测到：

best_start = 1735552

但直接从这里切 8 个 chirp 得到：

[508 500 477 504 454 470 53 491]

说明它不是好的 preamble 起点。

通过 local search 找到真正好的起点：

best_candidate_start = 1726208
peak bins = [507 507 507 507 507 507 507 507]

所以最终保存时应该使用：

best_candidate_start

而不是 coarse_start。

⸻

10. Local search 精确对齐

为了找到真正的 preamble 起点，在 coarse_start 附近搜索：

SEARCH_BEFORE = 20 * NSYM
SEARCH_AFTER = 1 * NSYM
SEARCH_STEP = 16

含义：

从 coarse_start 前面 20 个 chirp 开始搜索
搜索到 coarse_start 后面 1 个 chirp
每隔 16 samples 测试一个 candidate_start

每个 candidate_start 都截取：

4096 samples

然后 reshape 成：

8 × 512

对每个 chirp 分别计算 peak bin 和 peak ratio。

评分方式：

unique_8 = len(np.unique(peak_bins_8))
unique_first7 = len(np.unique(peak_bins_8[:7]))
unique_last7 = len(np.unique(peak_bins_8[1:]))
mean_ratio = np.mean(peak_ratios_8)
score = (unique_8, min(unique_first7, unique_last7), -mean_ratio)

评分含义：

第一优先：8 个 chirp 的 peak_bin 种类越少越好
第二优先：如果 8 个不完全一致，前 7 或后 7 越稳定越好
第三优先：如果稳定性一样，mean_ratio 越高越好

例如：

[507 507 507 507 507 507 507 507]

对应：

unique_8 = 1

这是最理想的情况。

⸻

11. 有效 preamble 的判断标准

当前实验中，我们只保留：

unique_8 == 1

也就是 8 个 chirp 的 peak bin 完全一致的样本。

例如：

[503 503 503 503 503 503 503 503]
[507 507 507 507 507 507 507 507]
[505 505 505 505 505 505 505 505]

这些都可以认为是高质量 preamble。

⸻

12. 保存数据

对于每个有效起点：

preamble = x[start:start + PREAMBLE_SAMPLES]

如果长度等于 4096，就加入数据集：

preambles.append(preamble)

最终转换为：

preambles = np.array(preambles, dtype=np.complex64)

保存：

preambles.tofile(OUT_FILE)

保存后的 shape 应该是：

(num_packets, 4096)

例如前 10 秒测试结果：

preambles shape = (8, 4096)
preambles dtype = complex64

⸻

13. 日志文件

为了方便检查每个 packet 的提取质量，将以下信息写入 .txt 日志文件：

coarse time
best time
score
unique_8
mean_ratio
bins

例如：

coarse time = 3.451584 s,
best time = 3.452416 s,
score = (1, 1, -26.124),
unique_8 = 1,
mean_ratio = 26.124,
bins = [507 507 507 507 507 507 507 507]

日志文件可以帮助检查：

每个 packet 是否成功提取
每个 preamble 的 8 个 peak bin 是否稳定
每个样本的 peak ratio 是否足够高

⸻

14. 当前阶段结果

在前 10 秒测试中，提取结果为：

valid preambles = 8
valid starts = [1205280 1726208 2247216 2767728 3288240 3808752 4329264 4849808]
valid times = [2.41056 3.452416 4.494432 5.535456 6.57648 7.617504 8.658528 9.699616]
preambles shape = (8, 4096)
preambles dtype = complex64

这说明：

前 10 秒中成功提取了 8 个高质量 8-upchirp preamble 样本。

⸻

15. 本阶段总结

本阶段完成了：

1. 生成本地 upchirp / downchirp
2. 读取 raw IQ 文件
3. 对单窗口做 dechirp + FFT
4. 计算 peak_bin 和 peak_ratio_db
5. 滑动窗口检测强 chirp
6. 根据 packet 间隔去重
7. 对每个 coarse_start 做 local search
8. 找到 8 个 peak bin 完全一致的 preamble 起点
9. 保存 8-upchirp complex IQ 样本
10. 写入日志文件检查提取质量

最终输出的数据可以用于下一步：

将每个 4096 点 complex IQ 样本转换成 spectrogram，
然后训练 spectrogram-based CNN。
---
你可以把这份笔记放在 `experiments/` 下面。  
代码文件可以放在：
```bash
scripts/extract_preamble_v2.py

笔记文件和代码文件分开，这样以后汇报、写论文、复现实验都会清楚很多。