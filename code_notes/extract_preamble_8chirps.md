## 代码理解
下面这份代码的核心目标是：

从原始 IQ 数据 iq_all2.bin 中，自动检测并提取 LoRa preamble 里的 8 个连续 upchirp，并保存成新的 bin 文件 node2_preambles.bin。

你可以把它理解成一个完整的 LoRa preamble extraction pipeline。

⸻

1. 基本参数设置

RAW_FILE = "/Users/zhaozhaoyan/Desktop/project-cnn/data/iq_all2.bin"
SF = 7
BW = 125e3
OS = 4
FS = BW * OS
M = 2 ** SF
NSYM = int(M * OS)
N_PREAMBLE = 8
PREAMBLE_SAMPLES = N_PREAMBLE * NSYM

这里定义了当前 LoRa 信号的基本参数。

实验参数是：

SF = 7
BW = 125 kHz
OS = 4
FS = BW × OS = 500 kS/s

其中：

M = 2^SF = 128

表示 LoRa 一个 symbol 的理论长度，也可以理解成 FFT bin 数。

因为采样率是带宽的 4 倍，所以一个 chirp 实际包含：

NSYM = M × OS = 128 × 4 = 512 samples

你要提取 8 个 upchirp，所以一个 preamble 片段长度是：

PREAMBLE_SAMPLES = 8 × 512 = 4096 complex IQ samples

也就是说，最后每一个提取出来的样本形状应该是：

1 个 preamble = 4096 个 complex64 IQ samples

⸻

2. 生成本地 upchirp 和 downchirp

def generate_upchirp(sf, bw, os_factor):
    fs = bw * os_factor
    M = 2 ** sf
    Ns = int(M * os_factor)
    t = np.arange(Ns) / fs
    T = Ns / fs
    k = bw / T
    phase = 2 * np.pi * ((-bw / 2) * t + 0.5 * k * t**2)
    upchirp = np.exp(1j * phase).astype(np.complex64)
    return upchirp

这个函数用来生成一个理想的 LoRa upchirp。

LoRa upchirp 是一个扫频信号，频率随着时间线性变化。这里的相位公式是：

phase = 2π [ (-BW/2)t + 0.5kt² ]

其中：

k = BW / T

表示扫频斜率。

然后通过：

upchirp = np.exp(1j * phase)

把相位转换成 complex IQ 信号。

在主程序中：

upchirp = generate_upchirp(SF, BW, OS)
downchirp = np.conj(upchirp)

这里 downchirp 是 upchirp 的共轭。

它的作用是做 dechirp。

简单来说：

接收到的 LoRa upchirp × 本地 downchirp ≈ 单频信号

所以，如果某个窗口真的包含 LoRa upchirp，那么 dechirp 之后再做 FFT，会出现一个明显的峰值。

⸻

3. 单个 chirp 窗口分析

def analyze_one_chirp_window(seg, downchirp):
    y = seg * downchirp
    Y = np.fft.fft(y)
    P = np.abs(Y) ** 2
    peak_bin = np.argmax(P)
    peak_power = np.max(P)
    mean_power = np.mean(P)
    peak_ratio_db = 10 * np.log10(peak_power / mean_power)
    return peak_bin, peak_ratio_db, peak_power, mean_power

这个函数是整个检测算法的核心。

输入：

seg：长度为 512 的 IQ 片段
downchirp：本地 downchirp

处理步骤：

y = seg * downchirp

这一步是 dechirp。

如果 seg 是 LoRa upchirp，那么 y 会变成近似单频信号。

然后：

Y = np.fft.fft(y)
P = np.abs(Y) ** 2

对 dechirp 后的信号做 FFT，并计算功率谱。

接着提取三个关键指标：

peak_bin = np.argmax(P)

表示 FFT 最大峰值所在的位置。

peak_power = np.max(P)

表示最大峰值功率。

mean_power = np.mean(P)

表示平均功率。

最后：

peak_ratio_db = 10 * np.log10(peak_power / mean_power)

计算 peak-to-average power ratio。

这个指标越大，说明 FFT 频谱里有一个越明显的主峰，也就越像 LoRa upchirp。

⸻

4. 检查一个 8-upchirp 片段

def evaluate_8chirp_start(x, candidate_start, downchirp):
    seg8 = x[candidate_start:candidate_start + PREAMBLE_SAMPLES]
    chirps = seg8.reshape(N_PREAMBLE, NSYM)
    peak_bins_8 = []
    peak_ratios_8 = []
    for i in range(N_PREAMBLE):
        seg = chirps[i]
        peak_bin, peak_ratio_db, peak_power, mean_power = analyze_one_chirp_window(
            seg,
            downchirp
        )
        peak_bins_8.append(peak_bin)
        peak_ratios_8.append(peak_ratio_db)
    peak_bins_8 = np.array(peak_bins_8)
    peak_ratios_8 = np.array(peak_ratios_8)
    return peak_bins_8, peak_ratios_8

这个函数的作用是：

假设 candidate_start 是一个 preamble 的起点，然后检查从这里开始的 8 个 chirp 是否合理。

它先截取：

seg8 = x[candidate_start:candidate_start + PREAMBLE_SAMPLES]

也就是 4096 个 IQ samples。

然后 reshape 成：

chirps = seg8.reshape(8, 512)

也就是：

第 1 行：第 1 个 upchirp
第 2 行：第 2 个 upchirp
...
第 8 行：第 8 个 upchirp

之后对每一个 chirp 分别做：

dechirp → FFT → 找 peak bin → 计算 peak ratio

最后返回：

peak_bins_8：8 个 chirp 的 FFT peak bin
peak_ratios_8：8 个 chirp 的 peak-to-average ratio

如果提取正确，理想情况下应该看到：

bins = [504 504 504 504 504 504 504 504]

也就是说，8 个 upchirp 的 peak bin 一致。

⸻

5. 在 coarse position 附近寻找最佳起点

def find_best_preamble_start(x, coarse_start, downchirp):
    SEARCH_BEFORE = 20 * NSYM
    SEARCH_AFTER = 1 * NSYM
    SEARCH_STEP = 16

这里的 coarse_start 是滑动窗口检测得到的粗略位置。

但是这个位置不一定刚好是 preamble 的真实起点，所以需要在它附近做 local search。

搜索范围是：

SEARCH_BEFORE = 20 * NSYM

表示向前搜索 20 个 chirp 的长度。

SEARCH_AFTER = 1 * NSYM

表示向后搜索 1 个 chirp 的长度。

因为 coarse detection 可能检测到的是 preamble 中间某个 chirp，所以真实起点可能在 coarse position 前面。因此这里向前搜索比较多。

搜索步长是：

SEARCH_STEP = 16

也就是说，每隔 16 个 samples 测试一次 candidate start。

⸻

5.1 评分标准 score

unique_8 = len(np.unique(peak_bins_8))
unique_first7 = len(np.unique(peak_bins_8[:7]))
unique_last7 = len(np.unique(peak_bins_8[1:]))
mean_ratio = np.mean(peak_ratios_8)
score = (unique_8, min(unique_first7, unique_last7), -mean_ratio)

这里的 score 用来判断哪个 candidate start 最好。

它是一个 tuple：

score = (unique_8, min(unique_first7, unique_last7), -mean_ratio)

Python 比较 tuple 的时候，会从左到右比较。

也就是说，优先级是：

第一优先级：unique_8 越小越好
第二优先级：first7 或 last7 的一致性越好
第三优先级：mean_ratio 越大越好

⸻

unique_8 是什么？

unique_8 = len(np.unique(peak_bins_8))

它表示 8 个 upchirp 的 peak bin 有几个不同值。

例如：

bins = [504 504 504 504 504 504 504 504]
unique_8 = 1

说明 8 个 chirp 的 peak bin 完全一致，这是最理想的情况。

如果是：

bins = [504 504 505 504 504 504 504 504]
unique_8 = 2

说明有一个 chirp 的 peak bin 和其他不一致，可能对齐不够好，或者受到噪声影响。

⸻

unique_first7 和 unique_last7 是什么？

unique_first7 = len(np.unique(peak_bins_8[:7]))
unique_last7 = len(np.unique(peak_bins_8[1:]))

这两个值分别检查：

前 7 个 chirp 是否一致
后 7 个 chirp 是否一致

为什么要这样做？

因为有时候第一个 chirp 或最后一个 chirp 可能因为边界、触发、噪声、对齐误差而不稳定。

比如：

bins = [503 504 504 504 504 504 504 504]

这时候：

unique_8 = 2
unique_first7 = 2
unique_last7 = 1

说明后 7 个 chirp 是稳定的，只有第一个有问题。

所以代码使用：

min(unique_first7, unique_last7)

来辅助判断这个 candidate start 是否仍然接近真实 preamble。

⸻

mean_ratio 是什么？

mean_ratio = np.mean(peak_ratios_8)

它表示 8 个 chirp 的平均 peak-to-average ratio。

例如：

mean_ratio = 26.496 dB

说明 8 个 chirp dechirp 后 FFT 主峰平均比背景能量高约 26 dB。

这个值越高，说明这些 chirp 的频域主峰越明显，检测越可靠。

⸻

为什么 score 里是 -mean_ratio？

因为代码用的是：

if best_score is None or score < best_score:

也就是 score 越小越好。

但是 mean_ratio 是越大越好。

所以写成：

-mean_ratio

这样 mean_ratio 越大，-mean_ratio 越小，score 就越好。

例如：

mean_ratio = 26
-score = -26

比：

mean_ratio = 20
-score = -20

更小，所以更好。

⸻

6. 读取原始 IQ 数据

x = np.fromfile(RAW_FILE, dtype=np.complex64)

这里从 iq_all2.bin 读取原始 IQ 数据。

因为你的数据是 complex64 格式，所以这里必须写：

dtype=np.complex64

如果 dtype 写错，后面的 sample 数、波形、FFT 结果都会出错。

然后打印：

print("Raw samples:", len(x))
print("Duration:", len(x) / FS, "s")
print("NSYM:", NSYM)
print("PREAMBLE_SAMPLES:", PREAMBLE_SAMPLES)

这些用于确认数据长度和参数是否正确。

⸻

7. 滑动窗口检测 chirp

HOP = 32
TEST_DURATION = 2100
TEST_SAMPLES = int(TEST_DURATION * FS)
THR_DB = 20
MIN_PACKET_GAP_S = 0.8

这里开始做粗检测。

参数含义：

HOP = 32

每次窗口向后移动 32 个 samples。

因为一个 chirp 是 512 samples，所以：

512 / 32 = 16

也就是一个 chirp 长度内会检测约 16 次。

TEST_DURATION = 2100

表示检测前 2100 秒的数据。

THR_DB = 20

表示如果 peak ratio 大于 20 dB，就认为这个窗口可能包含 chirp。

MIN_PACKET_GAP_S = 0.8

表示两个 packet 之间至少间隔 0.8 秒，用来去除重复检测。

⸻

7.1 滑动窗口循环

for start in range(0, TEST_SAMPLES - NSYM, HOP):
    seg = x[start:start + NSYM]
    peak_bin, peak_ratio_db, peak_power, mean_power = analyze_one_chirp_window(
        seg,
        downchirp
    )
    metrics.append(peak_ratio_db)
    starts.append(start)
    peak_bins.append(peak_bin)

这段代码从原始 IQ 数据开头开始，每次取 512 samples，做一次检测。

每个窗口都会得到：

peak_ratio_db：这个窗口像不像 chirp
peak_bin：FFT peak 的位置

最后保存到：

metrics：每个窗口的 peak ratio
starts：每个窗口的起始 sample index
peak_bins：每个窗口的 FFT peak bin

⸻

8. 选择 coarse candidate positions

high_idx = np.where(metrics > THR_DB)[0]

这一步找出所有 peak ratio 大于 20 dB 的窗口。

这些窗口都是候选 chirp 位置。

但是同一个 packet 附近会出现很多连续 high windows，所以不能全部当成 packet。

于是使用最小间隔去重：

min_gap = int(MIN_PACKET_GAP_S * FS)
candidate_coarse_starts = []
last_start = -10**12
for idx in high_idx:
    coarse_start = starts[idx]
    if coarse_start - last_start < min_gap:
        continue
    candidate_coarse_starts.append(coarse_start)
    last_start = coarse_start

这段代码的意思是：

如果当前 high window 距离上一个候选位置小于 0.8 秒，就认为它们属于同一个 packet，跳过。
如果距离大于 0.8 秒，就认为是新的 packet。

最后得到：

candidate_coarse_starts

也就是粗略的 packet 候选起点。

注意：这些还不是最终 preamble 起点，只是说明附近可能有 chirp。

⸻

9. 对每个 candidate 做精确对齐

valid_starts = []
valid_bins = []
valid_ratios = []
LOG_FILE = "/Users/zhaozhaoyan/Desktop/project-cnn/data/node2_preamble.txt"
log_f = open(LOG_FILE, "w", encoding="utf-8")

这里创建三个列表：

valid_starts：保存最终有效 preamble 的起点
valid_bins：保存每个有效 preamble 的 8 个 peak bin
valid_ratios：保存每个有效 preamble 的 8 个 peak ratio

同时打开日志文件 node2_preamble.txt，用来记录检测结果。

⸻

9.1 对每个 coarse candidate 做 local search

for coarse_start in candidate_coarse_starts:
    best_candidate_start, best_bins, best_ratios, best_score = find_best_preamble_start(
        x,
        coarse_start,
        downchirp
    )

这一步对每个粗略位置进行精细搜索，找到最优的 8-upchirp 起点。

返回：

best_candidate_start：最优 preamble 起点
best_bins：该位置下 8 个 chirp 的 peak bin
best_ratios：该位置下 8 个 chirp 的 peak ratio
best_score：评分

然后：

unique_8 = len(np.unique(best_bins))
mean_ratio = np.mean(best_ratios)

计算 8 个 chirp 的一致性和平均检测强度。

⸻

9.2 写入日志

log_line = (
    f"coarse time = {coarse_start / FS:.6f} s, "
    f"best time = {best_candidate_start / FS:.6f} s, "
    f"score = {best_score}, "
    f"unique_8 = {unique_8}, "
    f"mean_ratio = {mean_ratio:.3f}, "
    f"bins = {best_bins}\n"
)

日志里每一行的含义是：

coarse time：
滑动窗口粗检测得到的时间。
best time：
local search 后找到的最佳 preamble 起点时间。
score：
该 candidate 的评分，越小越好。
unique_8：
8 个 upchirp 的 peak bin 有几个唯一值。
mean_ratio：
8 个 upchirp 的平均 peak-to-average ratio。
bins：
8 个 upchirp 分别对应的 FFT peak bin。

例如：

coarse time = 33.656512 s,
best time = 33.657312 s,
score = (1, 1, np.float32(-26.495506)),
unique_8 = 1,
mean_ratio = 26.496,
bins = [504 504 504 504 504 504 504 504]

这说明：

这个 packet 被检测到了；
最佳起点在 33.657312 s；
8 个 upchirp 的 peak bin 全部是 504；
unique_8 = 1，说明完全一致；
mean_ratio = 26.496 dB，说明 FFT 主峰很明显。

⸻

9.3 只保留有效 preamble

if unique_8 == 1:
    valid_starts.append(best_candidate_start)
    valid_bins.append(best_bins)
    valid_ratios.append(best_ratios)

这里的有效标准是：

unique_8 == 1

也就是说，只有当 8 个 chirp 的 peak bin 完全一致时，才保存这个 preamble。

这是一个比较严格的标准。

它可以保证你提取出来的 preamble 对齐质量比较高。

⸻

10. 保存 summary

valid_starts = np.array(valid_starts)
valid_bins = np.array(valid_bins)
valid_ratios = np.array(valid_ratios)

把列表转成 numpy array。

然后写 summary：

summary_text = (
    "\n===== Summary =====\n"
    f"Valid preambles: {len(valid_starts)}\n"
    f"Valid starts: {valid_starts}\n"
    f"Valid times: {valid_starts / FS}\n"
)

summary 里会记录：

Valid preambles：
最终有效 preamble 数量。
Valid starts：
每个有效 preamble 在原始 IQ 数据中的 sample index。
Valid times：
每个有效 preamble 对应的时间，单位秒。

最后关闭日志文件：

log_f.close()

⸻

11. 提取并保存有效的 8-upchirp preambles

OUT_FILE = "/Users/zhaozhaoyan/Desktop/project-cnn/data/node2_preambles.bin"
preambles = []
for start in valid_starts:
    preamble = x[start:start + PREAMBLE_SAMPLES]
    if len(preamble) != PREAMBLE_SAMPLES:
        continue
    preambles.append(preamble)

这一步根据 valid_starts，从原始 IQ 数据中真正截取 8 个 upchirp。

每个 preamble 长度是：

4096 complex IQ samples

然后：

preambles = np.array(preambles, dtype=np.complex64)

得到一个二维数组：

preambles.shape = (有效 preamble 数量, 4096)

例如如果提取了 2000 个 preamble：

preambles.shape = (2000, 4096)

然后保存成 bin 文件：

preambles.tofile(OUT_FILE)

这个文件就是后面 CNN 可以使用的数据源之一。

⸻

12. 读回检查

preambles_read = np.fromfile(OUT_FILE, dtype=np.complex64)
print("\nRead back samples:", len(preambles_read))
preambles_read = preambles_read.reshape(-1, PREAMBLE_SAMPLES)
print("Read back shape:", preambles_read.shape)
print("Same number of preambles:", preambles_read.shape[0] == preambles.shape[0])
print("Same sample length:", preambles_read.shape[1] == PREAMBLE_SAMPLES)

这一步是为了确认保存的 bin 文件没有问题。

先重新读取：

preambles_read = np.fromfile(OUT_FILE, dtype=np.complex64)

然后 reshape：

preambles_read.reshape(-1, 4096)

如果保存正确，应该看到：

Same number of preambles: True
Same sample length: True

这说明：

保存的 preamble 数量正确；
每个 preamble 的长度正确；
bin 文件可以正常用于后续处理。

⸻

整体流程总结

你这段代码可以总结成下面这个流程：

1. 设置 LoRa 参数：SF=7, BW=125 kHz, FS=500 kS/s
2. 计算一个 upchirp 长度：512 samples
3. 生成本地 upchirp 和 downchirp
4. 读取原始 IQ 文件 iq_all2.bin
5. 用 512-sample 滑动窗口扫描原始信号
6. 对每个窗口做 dechirp + FFT
7. 用 peak-to-average power ratio 检测可能的 chirp 位置
8. 用 0.8 秒最小间隔去除重复 packet 检测
9. 对每个 coarse candidate 做 local search
10. 每个 candidate start 截取 4096 samples
11. 分成 8 个 chirp，每个 chirp 做 dechirp + FFT
12. 检查 8 个 chirp 的 FFT peak bin 是否一致
13. 如果 unique_8 = 1，则认为提取成功
14. 保存有效 preamble 到 node2_preambles.bin
15. 重新读取 bin 文件，确认保存正确

⸻

这段代码的核心判断逻辑

最重要的判断是这一句：

if unique_8 == 1:

它表示：

如果 8 个 upchirp 的 FFT peak bin 完全一致，
则认为这个 preamble 提取成功。

例如：

bins = [506 506 506 506 506 506 506 506]
unique_8 = 1
mean_ratio = 25.9 dB

这说明：

8 个 chirp 都在同一个 FFT bin 出现主峰；
dechirp 后表现稳定；
这个片段很可能就是正确对齐的 LoRa preamble。

所以你的 txt 日志和 heatmap 图就是用来证明这一点的。

