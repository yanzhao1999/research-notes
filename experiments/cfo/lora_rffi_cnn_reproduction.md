# LoRa RFFI CNN Reproduction Notes
## 1. Data Collection and Input Samples
本实验的目标是复现 LoRa RFFI 论文中的 CNN-based device classification 方法。当前阶段使用两个 LoRa 节点采集原始 IQ 数据，并基于提取出的 LoRa preamble 构建后续 CNN 输入样本。
实验中使用的主要 LoRa 物理层参数为：
- **Spreading Factor**: SF7
- **Bandwidth**: 125 kHz
- **Sampling rate**: 500 kS/s
### 1.1 LoRa chirp duration
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
因此，一个 LoRa chirp 的时间长度为：

T_s = 1.024 ms

### 1.2 Oversampling factor

接收端采样率为：

$$
F_s = 500,kS/s
$$

LoRa 信号带宽为：

$$
BW = 125,kHz
$$

因此，过采样因子为：

$$
OS = \frac{F_s}{BW}
= \frac{500000}{125000}
= 4
$$

过采样因子可以理解为：接收端实际采样率比 LoRa 信号带宽对应的基础采样率高了 4 倍。这样可以获得更多采样点，使 chirp 形状更清楚，也有利于后续的同步、dechirp 和 FFT 检测。

### 1.3 Number of samples per chirp

一个 LoRa chirp 对应的采样点数为：

$$
N_{chirp} = 2^{SF} \times OS
= 128 \times 4
= 512
$$

因此，在当前实验配置下，一个 LoRa upchirp 包含：

512 complex IQ samples

### 1.4 Extracted preamble samples

在数据预处理阶段，我没有使用完整 LoRa packet，而是只提取 packet preamble 中的 8 个连续 upchirps。

由于每个 upchirp 包含 512 个 IQ samples，所以每个提取样本的长度为：

$$
N_{packet} = 8 \times 512 = 4096
$$

因此，每个样本包含：

4096 complex IQ samples

**详细的 8-upchirp 提取过程记录在：**

**extract_preamble_8chirps.md** 

⸻

### 2. IQ Samples

IQ 数据是无线信号的复数基带表示。

I = In-phase component，同相分量
Q = Quadrature component，正交分量

每一个 IQ sample 可以写成：

$$
x[n] = I[n] + jQ[n]
$$

可以把一个 IQ sample 理解为复平面上的一个点：

        Q
        ↑
        |
        |      • one IQ sample
        |
        |
--------+----------------→ I

其中：

* 点到原点的距离表示信号幅度；
* 点相对于 I 轴的角度表示信号相位。

通俗地说，IQ 数据就是无线信号随时间变化的二维坐标记录。
如果采样率为 500 kS/s，表示接收机每秒记录 500,000 个 IQ samples。

⸻

### 3. Preamble Extraction

### 3.1 Dechirp and FFT detection

为了检测 LoRa upchirp，我首先在本地生成一个理想的 LoRa upchirp，并取其共轭得到 downchirp。

对于接收信号中的每一个长度为 512 samples 的窗口，将其与本地 downchirp 相乘：

$$
y[n] = r[n] \cdot d^*[n]
$$

其中：

* $r[n]$ 表示接收信号窗口；
* $d^*[n]$ 表示本地 upchirp 的共轭信号，也就是 downchirp。

如果该窗口中确实包含 LoRa upchirp，那么经过 dechirp 后，原本的扫频信号会被转换为近似单频信号。此时，对 dechirped signal 做 FFT，频谱中应该出现明显峰值。

检测指标定义为 FFT 最大功率与平均功率之间的比值：

$$
M = 10\log_{10}\left(\frac{\max |Y[k]|^2}{\mathrm{mean}(|Y[k]|^2)}\right)
$$

当该指标大于设定阈值时，认为该窗口附近可能存在 LoRa upchirp，从而得到一个粗略的 packet 起点位置。


### 4. FFT Verification

在提取出 8 个 upchirps 后，我使用 dechirp + FFT 对提取结果进行验证。

对于每个提取出的 packet：

8 upchirps × 512 samples = 4096 complex IQ samples

我将其分成 8 段，每段对应一个 upchirp。然后对每个 upchirp 分别进行 dechirp 和 FFT，并记录其 FFT peak bin。

如果 8 个 upchirps 是正确提取并且对齐的，那么它们 dechirp 后的 FFT peak bin 应该保持一致。

例如：

bins = [504 504 504 504 504 504 504 504]
unique_8 = 1

其中：

unique_8 = 1

表示 8 个 upchirps 的 FFT peak bin 只有一个唯一值，说明这 8 个 upchirps 的频域特征一致，preamble 提取和对齐是稳定的。

⸻


### 6. Dataset Construction

### 6.1 NumPy dataset format

.npy 是 NumPy 用来保存数组的文件格式。在本实验中，它用于保存已经处理好的输入数据 X 和标签 y，方便后续 CNN 训练时直接读取。

例如，对于 IQ-based CNN，数据可以保存为：

X_iq_2nodes.npy
y_2nodes.npy

其中：

X.shape = (4000, 4096, 2)

表示：

* 4000 个 samples；
* 每个 sample 有 4096 个 IQ 点；
* 每个 IQ 点包含 I 和 Q 两个通道。

标签：

y.shape = (4000,)

表示 4000 个样本对应的类别标签。

### 6.2 IQ normalization

在构建 CNN 数据集后，我对每个 packet 进行了 RMS normalization。

原因是不同 packet 的接收幅度可能受到距离、天线方向、接收增益和信道条件等因素影响。如果不进行归一化，CNN 可能会主要根据接收信号强度进行分类，而不是学习真正的射频指纹特征。

归一化公式对应代码为：

power = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
X = X / (power + 1e-12)

其作用是将每个 packet 的平均能量归一化到相近水平，从而减少幅度差异对分类结果的影响。

⸻
### 7. CNN Reproduction

本实验当前完成的是 LoRa RFFI 论文中 IQ-based CNN device classification 方法的复现。

实现流程包括：

8-upchirp preamble extraction
→ dechirp + FFT verification
→ IQ dataset construction
→ train / validation / test split
→ IQ-based CNN training
→ model saving
→ separate test evaluation

CNN 的输入为：

X.shape = (4000, 4096, 2)

其中 4096 表示每个样本包含 8 个 upchirps，每个 upchirp 包含 512 个 IQ samples；2 表示 I/Q 两个通道。

8. Results

当前 spectrogram-based CNN 在两个 LoRa 节点数据集上的实验结果为：

Train accuracy = 100%
Validation accuracy = 100%
Test accuracy = 100%
Test loss ≈ 0.0011

测试集包含：

node1: 1000 samples
node2: 1000 samples
total: 2000 samples

该结果说明，在当前同一实验条件下，spectrogram 特征能够有效区分两个 LoRa 节点。

需要注意的是，该结果目前只能说明模型在当前同一批数据上表现很好，还不能证明其在跨天、不同环境或 CFO drift 条件下仍然稳定。

⸻

# Spectrogram-based CNN Reproduction

可以，下面是我帮你整理后的更正式、更顺、更适合放进实验笔记/README 的版本。我把重复的地方合并了，把 FFT/STFT 那一段放到了更合适的位置，并且统一了编号和逻辑。

⸻

Spectrogram-based CNN 实验记录

### 1. 实验目标

本实验的目标是复现 LoRa RFFI 论文中的 spectrogram-based CNN 设备分类方法。

在前面的实验中，我已经完成了 LoRa preamble 的提取，并得到了两个 LoRa 节点对应的 preamble binary files。因此，本实验不再从原始 IQ 采集文件开始，而是直接使用已经提取好的 8-upchirp preamble samples 作为输入。

本实验的主要流程为：

提取好的 preamble bin 文件
→ STFT 生成 spectrogram
→ 保存 node1 / node2 spectrogram 数据
→ 合并两个节点的数据
→ 创建标签
→ 划分训练集、验证集和测试集
→ 打乱训练集
→ 训练 spectrogram-based CNN
→ 测试集评估

⸻

### 2. 输入数据

本实验的输入是已经提取好的 LoRa preamble bin 文件。

每个 LoRa 节点对应一个 preamble 文件：

node1 preamble bin file
node2 preamble bin file

每个 preamble sample 包含 8 个连续的 LoRa upchirps。

当前实验使用的 LoRa 物理层参数为：

SF = 7
BW = 125 kHz
Sampling rate = 500 kS/s
Oversampling factor = 4

在该配置下，一个 LoRa upchirp 包含：

512 complex IQ samples

因此，一个 8-upchirp preamble sample 包含：

8 × 512 = 4096 complex IQ samples

也就是说，每一个输入样本都是一段长度为 4096 的 complex IQ signal。

⸻

### 3. FFT 与 STFT 的区别

在本实验中，FFT 和 STFT 都与频域分析有关，但它们的用途不同。

3.1 FFT

FFT 是对一整段信号做一次频域分析。

例如，一个 upchirp 的长度是 512 samples，对该 upchirp 做 FFT 后，可以得到 512 个 frequency bins。

FFT 可以告诉我们这整段信号中包含哪些频率成分，但是不能显示这些频率成分如何随时间变化。

简单来说：

FFT = 只看整体频率成分

在前面的 preamble extraction 阶段，我主要使用 dechirp + FFT 来验证 8 个 upchirps 是否提取正确并且对齐稳定。

如果 8 个 upchirps 经过 dechirp 后的 FFT peak bin 保持一致，说明提取出的 preamble 比较稳定。

⸻

3.2 STFT

STFT 是 Short-Time Fourier Transform，即短时傅里叶变换。

它的基本思想是：

把一段较长的信号切成多个短窗口
→ 对每个短窗口分别做 FFT
→ 得到每个时间位置上的频率分布
→ 最终形成二维时频表示

例如，对于一个 4096-sample 的 8-upchirp preamble，可以设置：

nperseg = 256
noverlap = 128
nfft = 256

含义是：

每个 STFT 窗口长度 = 256 IQ samples
相邻窗口重叠 = 128 IQ samples
每个窗口做 256 点 FFT

STFT 的输出是一个二维时频矩阵：

Zxx.shape = (frequency bins, time frames)

简单来说：

STFT = 看频率成分如何随时间变化

对于 LoRa chirp 来说，STFT 可以更直观地展示 chirp 的扫频结构，因此可以用于生成 spectrogram，并作为 CNN 的输入特征。

⸻

### 4. STFT 与 Spectrogram 生成

本实验的第一步是对已经提取好的 preamble samples 进行 STFT。

对于每一个 4096-sample 的 preamble：

输入：4096 complex IQ samples
输出：一个 spectrogram

Spectrogram 是由 STFT 得到的二维时频表示。它可以表示信号频率成分随时间的变化情况。

由于 LoRa 信号基于 Chirp Spread Spectrum, CSS，chirp 的频率会随时间扫动，因此 spectrogram 能够比原始 IQ 序列更直观地表示 LoRa preamble 的时频结构。

在本实验中，每个 preamble sample 都会被转换成一个 spectrogram。

⸻

### 5. Node1 和 Node2 的 Spectrogram 数据

对两个 LoRa 节点分别进行 spectrogram 生成。

对于 node1：

node1 preamble samples
→ STFT
→ node1 spectrogram samples

对于 node2：

node2 preamble samples
→ STFT
→ node2 spectrogram samples

生成完成后，分别保存为两个 NumPy 数据文件：

node1_spectrograms.npy
node2_spectrograms.npy

其中，每个 .npy 文件保存的是该节点所有 preamble samples 对应的 spectrogram 数据。

如果每个节点有 2000 个 preamble samples，那么：

node1_spectrograms.npy 包含 2000 个 spectrogram samples
node2_spectrograms.npy 包含 2000 个 spectrogram samples

⸻

### 6. 数据集合并

生成 node1 和 node2 的 spectrogram 数据后，需要将两个节点的数据合并成一个完整的数据集。

合并前：

node1_spectrograms.npy
node2_spectrograms.npy

合并后得到：

X_spectrograms

其中：

前 2000 个 samples 来自 node1
后 2000 个 samples 来自 node2

因此，如果每个节点有 2000 个样本，则合并后的数据集共有：

4000 个 spectrogram samples

这个合并后的数据集作为 CNN 的输入数据。

⸻

### 7. 标签创建

为了进行监督学习，需要为每个 spectrogram sample 创建对应的标签。

本实验中使用两个类别：

label 0 → node1
label 1 → node2

因此：

node1 的 2000 个 samples 对应 2000 个 label 0
node2 的 2000 个 samples 对应 2000 个 label 1

最终标签向量为：

前 2000 个标签为 0
后 2000 个标签为 1

这样，CNN 在训练时就可以学习每个 spectrogram sample 与对应设备类别之间的关系。

⸻

### 8. 训练集、验证集和测试集划分

完成数据合并和标签创建之后，需要将数据集划分为训练集、验证集和测试集。

本实验中，每个节点有 2000 个 samples。划分方式为：

每个节点前 1000 个 samples 用于 training / validation
每个节点后 1000 个 samples 用于 test

对于每个节点的前 1000 个 samples：

900 个 samples 用于 training
100 个 samples 用于 validation

因此，最终数据划分为：

Training set:
node1: 900 samples
node2: 900 samples
total: 1800 samples
Validation set:
node1: 100 samples
node2: 100 samples
total: 200 samples
Test set:
node1: 1000 samples
node2: 1000 samples
total: 2000 samples

这种划分方式可以保证训练集、验证集和测试集之间相互独立。测试集不参与模型训练，只用于评估模型在未见过数据上的分类能力。

⸻

### 9. 训练集打乱

在训练 CNN 之前，只对训练集进行随机打乱。

原因是合并后的训练数据可能按照类别顺序排列：

先是 node1 的 samples
后是 node2 的 samples

如果不打乱训练集，模型在训练过程中会先连续看到大量 node1 数据，再连续看到大量 node2 数据，这可能会影响训练稳定性。

因此，在训练前需要随机打乱 training set，使 node1 和 node2 的样本混合出现。

需要注意的是：

只打乱 training set
validation set 和 test set 保持不变

这样既可以提高训练过程的稳定性，又可以保证验证集和测试集的评估结果固定、可复现。

⸻

### 10. Spectrogram-based CNN 训练

完成数据准备之后，将 spectrogram 作为 CNN 的输入进行训练。

CNN 的输入是二维 spectrogram 图像。模型通过卷积层自动学习 spectrogram 中与不同 LoRa 设备相关的特征。

对于本实验中的两个 LoRa 节点，CNN 的任务是进行二分类：

class 0: node1
class 1: node2

训练过程中，模型根据 spectrogram 和对应标签不断更新参数，使其能够正确区分两个节点。

训练完成后，保存训练好的 CNN 模型，用于后续测试和分析。

⸻

### 11. 测试集评估

模型训练完成后，使用独立的 test set 进行评估。

测试集包含：

node1: 1000 samples
node2: 1000 samples
total: 2000 samples

这些测试样本没有参与模型训练，因此可以用于评估模型对未见过数据的分类能力。

测试阶段主要观察以下指标：

test accuracy
test loss
confusion matrix
classification report

其中：

* test accuracy 用于表示模型在测试集上的整体分类正确率；
* test loss 用于表示模型在测试集上的损失值；
* confusion matrix 用于观察 node1 和 node2 之间是否存在混淆；
* classification report 用于进一步分析 precision、recall 和 F1-score。

如果测试准确率较高，同时 confusion matrix 中错误分类数量较少，说明模型能够从 spectrogram 特征中有效区分两个 LoRa 节点。

⸻

### 12. 当前实验结果

当前 spectrogram-based CNN 在两个 LoRa 节点数据集上的实验结果为：

Train accuracy = 100%
Validation accuracy = 100%
Test accuracy = 99.95%
Test loss ≈ 0.0023

测试集包含 2000 个 samples：

node1: 1000 samples
node2: 1000 samples

2000 个测试样本中，1999 个分类正确，1 个分类错误。
最终测试准确率为 99.95%。

实验结果说明，在当前同一实验条件下，spectrogram-based CNN 能够非常准确地区分两个 LoRa 节点。

这表明 spectrogram 中确实包含了与设备差异相关的特征，CNN 可以从这些时频特征中学习到分类信息。

⸻




## Hybrid Classifier

Hybrid Classifier 是一种结合 **CNN softmax probability** 和 **CFO 物理层约束** 的分类后处理方法。首先，CNN 对待分类样本输出概率向量 \(S\)，其中 \(S_k\) 表示该样本属于第 \(k\) 个设备的概率。然后，将该样本的 CFO 估计值 \(\Delta \hat{f}_{DUT}\) 与数据库中第 \(k\) 个设备的 reference CFO \(\Delta \hat{f}_k\) 进行比较。如果

\[
|\Delta \hat{f}_{DUT} - \Delta \hat{f}_k| > \lambda
\]

则认为该样本不太可能来自第 \(k\) 个设备，因此将对应概率置为：

\[
S_k = 0
\]

如果 CFO 差值没有超过阈值，则保留原始 CNN 概率。最后，从更新后的 \(S\) 中选择概率最大的设备作为最终预测结果。
## 例子

在本实验中，两个节点的 reference CFO 分别为：

node1_ref_cfo = -7.27795 bins
node2_ref_cfo = -4.5995 bins

对于测试样本 index 72，CNN 的 softmax 输出为：

S = [0.1852, 0.8148]

这表示 CNN 认为该样本属于 node1 的概率为 0.1852，属于 node2 的概率为 0.8148。因此，如果只使用 CNN，最终会预测为 node2。

但是该样本的 CFO 为：

CFO_DUT = -7.2455 bins

它与两个 reference CFO 的距离分别为：

Distance to node1_ref = |-7.2455 - (-7.27795)| = 0.03245
Distance to node2_ref = |-7.2455 - (-4.5995)| = 2.646

如果设置：

λ = 0.8 bins

那么 node1 的 CFO 距离小于阈值，因此 node1 的概率保留；node2 的 CFO 距离大于阈值，因此 node2 的概率被置为 0。于是：

Before CFO filtering:
S = [0.1852, 0.8148]
After CFO filtering:
S = [0.1852, 0]

最终 Hybrid Classifier 选择 node1 作为预测结果。这个例子说明，虽然 CNN 原本将该样本误判为 node2，但 CFO 特征显示它更接近 node1，因此 Hybrid Classifier 成功修正了 CNN 的错误分类。





