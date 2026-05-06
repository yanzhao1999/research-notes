# LoRa RFFI CNN Reproduction Notes

## 1. Data Collection
原始 IQ 数据采集与输入样本构建

首先，我使用两个 LoRa 节点分别采集原始 IQ 数据，用于后续的射频指纹识别实验。实验中 LoRa 的主要物理层参数设置为：

- **Spreading Factor**：SF7
- **Bandwidth**：125 kHz
- **Sampling rate**：500 kS/s

##### 1. LoRa chirp 的时间长度

在 LoRa 中，一个 chirp symbol 的理论持续时间由 SF 和带宽决定：

$$
T_s = \frac{2^{SF}}{BW}
$$

当 $SF=7$，$BW=125\,kHz$ 时：

$$
T_s = \frac{2^7}{125000}
= \frac{128}{125000}
= 1.024\,ms
$$

也就是说，一个 LoRa chirp 的时间长度为 **1.024 ms**。

### 2. 过采样因子计算
os过采样因子，可以理解为接收机实际采样得比 LoRa 信号本身需要的采样率多了几倍。这样做的好处是

**采样点更多**
**时间分辨率更高**
**chirp 形状更清楚**
**dechirp 和 FFT 更稳定**
**同步和检测更方便**

由于接收端采样率设置为：

$$
F_s = 500\,kS/s
$$

而 LoRa 信号带宽为：

$$
BW = 125\,kHz
$$

所以过采样因子为：

$$
OS = \frac{F_s}{BW}
= \frac{500000}{125000}
= 4
$$

### 3. 每个 chirp 的采样点数

因此，一个 LoRa chirp 对应的采样点数为：

$$
N_{chirp} = 2^{SF} \times OS
= 128 \times 4
= 512
$$

也就是说，在当前实验配置下，每一个 LoRa preamble upchirp 都包含 **512 个 IQ samples**。

### 4. 提取 8 个 preamble upchirps

在数据预处理阶段，我没有使用整个 LoRa packet，而是只提取 packet 前导码中的 **8 个连续 upchirps**。
由于每个 upchirp 包含 **512 个 IQ samples**，而每个 packet 提取 **8 个 upchirps**，所以每个 packet 对应的 IQ 数据长度为：

$$
N_{packet} = 8 \times 512 = 4096
$$

### 5. IQ samples的理解
I 是 In-phase，意思是同相分量。
Q 是 Quadrature，意思是正交分量。
IQ 数据可以理解为无线信号的复数基带表示。每一个 IQ sample 由 I 和 Q 两个分量组成，其中 I 类似横坐标，Q 类似纵坐标。它们合在一起可以描述信号在某一时刻的幅度和相位。**通俗的认为** IQ 数据就是无线信号的二维坐标记录。I 是横坐标，Q 是纵坐标。一个 IQ sample 就是一个点。很多 IQ samples 连起来，就描述了无线信号随时间的变化。
一个 IQ sample = 平面上的一个点。这个点包含两个信息：
离原点多远 → 信号强度
转到哪个角度 → 信号相位

        Q
        ↑
        |
        |      • 这个点就是一个 IQ sample
        |
        |
--------+----------------→ I

**sample**解释
sample = 采样点
sampling rate = 每秒采多少个点
500 kS/s = 每秒采 500,000 个点
IQ sample = 每个采样点包含 I 和 Q 两个值
## 2. Preamble Extraction

**单频信号 = 只有一个固定频率的波。**

### Dechirp 与 FFT 检测

为了检测 LoRa upchirp，我首先在接收端本地生成一个理想的 LoRa upchirp，并取其共轭得到 downchirp。对于接收信号中的每一个长度为 512 samples 的窗口，将其与 downchirp 相乘：

$$
y[n] = r[n] \cdot d^*[n]
$$

其中 $r[n]$ 表示接收信号窗口，$d^*[n]$ 表示本地 upchirp 的共轭信号。

如果该窗口中存在 LoRa upchirp，那么经过 dechirp 后，原本的扫频信号会被转换为近似单频信号。因此，对 dechirp 后的信号进行 FFT 时，频谱中会出现明显峰值。

检测指标定义为 FFT 最大功率与平均功率之间的比值：

$$
M = 10\log_{10}\left(\frac{\max |Y[k]|^2}{\mathrm{mean}(|Y[k]|^2)}\right)
$$

当该指标大于设定阈值时，认为该窗口附近可能存在 LoRa upchirp，从而得到一个粗略的 packet 起点位置。

### 局部精细搜索

由于粗检测阶段的滑动步长为 32 samples，因此检测到的位置并不一定与真实 preamble 起点完全对齐。为了提高截取精度，我在粗略起点附近进行局部搜索。

具体来说，在 coarse start 前 1024 samples 到后 512 samples 的范围内，每隔 16 samples 尝试一个候选起点。对于每个候选起点，截取连续 4096 samples，并将其划分为 8 个 chirps。然后分别对每个 chirp 进行 dechirp 和 FFT，记录每个 chirp 的 FFT peak bin。

由于 LoRa preamble 由重复的 upchirp 组成，如果候选起点对齐正确，则 8 个 chirp 的 FFT peak bin 应该保持一致。因此，我使用 8 个 chirp 的 peak bin 一致性作为评价标准。评分方式为：

$$
score = (N_{unique,8}, N_{unique,7})
$$

其中 $N_{unique,8}$ 表示 8 个 chirp 的 peak bin 中不同取值的数量，$N_{unique,7}$ 表示忽略第一个 chirp 后，后 7 个 chirp 的 peak bin 中不同取值的数量。

选择 score 最小的位置作为最终 preamble 起点。这样可以尽量保证截取到的 8 个 upchirps 在时间上对齐，并提高后续 FFT 特征提取和 CNN 训练数据的稳定性。

### 提取结果质量检查

最后，我对所有提取出的 packet 进行质量检查。对于每个 packet，统计其 8 个 chirp 的 FFT peak bin 是否完全一致，同时也统计忽略第一个 chirp 后，后 7 个 chirp 的 peak bin 是否一致。

如果大多数 packet 的 8-chirp 或 7-chirp peak bin 保持稳定，则说明 preamble 提取结果较为可靠。

**核心**
**我利用 LoRa upchirp 在 dechirp 后会变成单频信号的特性，通过 FFT peak 检测 chirp 位置；然后利用 preamble 中多个 upchirp 重复一致的特点，在粗检测位置附近搜索，使 8 个 chirp 的 FFT peak bin 尽量一致，从而得到更准确的 preamble 起点。**

## 3. FFT Verification

## 4. Dataset Construction

**npy** 是 NumPy 用来保存数组的文件格式。
在你的实验中，它的作用是把已经处理好的 IQ 数据 X 和标签 y 保存下来，方便后面 CNN 训练时直接读取使用。
举个例子就是，我保存了2个npy文件，分别是X_iq_2nodes.npy和
y_2nodes.npy，然后X里的是CNN 的输入数据，也就是 IQ 数据。**shape** = 数据的尺寸，他的形状是X.shape = (4000, 4096, 2)，意思是4000 个 samples，每个 sample 有 4096 个 IQ 点，每个 IQ 点有 I 和 Q 两个通道。然后另一个y保存的是标签，它的形状是(4000,)，也就是代表4000个label标签。


**IQ 数据归一化**
在构建 CNN 数据集后，我对每个 packet 进行了 RMS normalization。由于不同 packet 的接收幅度可能受到距离、天线方向、接收增益和信道条件等因素影响，如果不进行归一化，CNN 可能会主要根据接收信号强度进行分类，而不是学习真正的射频指纹特征。

归一化代码如下：
python
power = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
X = X / (power + 1e-12)

## 5. CNN Reproduction

## 6. Results

## 7. Next Steps