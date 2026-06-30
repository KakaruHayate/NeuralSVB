# NeuralSVB — 环境搭建与运行指南

> 基于 [MoonInTheRiver/NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB) 的复现修复版。
> 修复了 Python 3.12 / torch 2.x / numpy 2.x / librosa 0.11+ 兼容性及 Windows 多进程问题。

---

## 目录结构

```
J:/NSVB/
├── repo/                        # 克隆的 NeuralSVB 仓库
├── .venv/                       # UV Python 虚拟环境
├── data/
│   ├── raw/PopBuTFy/            # 原始数据集 (MP3 + TXT)
│   │   └── data/                # 904 个歌手/歌曲子目录
│   ├── processed/
│   │   └── PopBuTFy_new/
│   │       └── spk_emb/         # 说话人嵌入 (.npy)
│   └── binary/
│       └── PopBuTFyENSpkEM_new/  # 二值化训练数据
├── checkpoints/
│   ├── 1009_pretrain_asr_english/  # ASR PPG 提取器
│   ├── 1012_hifigan_all_songs_nsf/ # HiFi-GAN 声码器
│   └── vae_mle/                   # VAE MLE 主模型
├── *.zip                        # 原始物料
├── scan_imports.py              # 依赖扫描
├── save_spkemb_single.py        # 单进程嵌入提取
├── bin_para_single.py           # 单进程二值化
├── PATCH_SUMMARY.md             # 代码修改清单
└── .venv/                       # Python 虚拟环境
```

## 一、环境准备

### 前置条件
- Python ≥ 3.10 (推荐 3.12)
- CUDA-capable GPU (可选，CPU 推理极慢)
- 磁盘空间 ≥ 15GB

### 安装 UV (包管理器)

```bash
# Windows
pip install uv
# 或
curl -fsSL https://astral.sh/uv/install.sh | sh
```

### 创建虚拟环境

```bash
cd /path/to/NeuralSVB
uv venv --python 3.12 .venv
```

### 安装依赖

```bash
# 安装 PyTorch (CUDA 12.4 版本)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装其余依赖
uv pip install numpy scipy scikit-learn einops pyyaml tqdm h5py \
  librosa soundfile resampy pyworld pysptk praat-parselmouth \
  resemblyzer pycwt transformers tokenizers tensorboardX matplotlib \
  pandas packaging chardet six jieba nltk g2p-en pypinyin \
  webrtcvad pyloudnorm Cython scikit-image g2pm tslearn torchcrf
```

> **注意**: `torchcrf` 可能因 C++ 编译失败，但推理不需要。

### 验证安装

```bash
.venv/Scripts/python.exe -c "
import torch, librosa, numpy, resemblyzer, pyworld, parselmouth
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('librosa:', librosa.__version__)
print('ok')
"
```

## 二、物料准备

### 下载

| 物料 | 来源 |
|------|------|
| PopBuTFy.zip (数据集) | 联系作者或 Google Drive |
| PopBuTFy-preview.zip | GitHub Releases |
| text_labels.zip | GitHub Releases |
| 1009_pretrain_asr_english.zip | GitHub Releases |
| 1012_hifigan_all_songs_nsf.zip | GitHub Releases |
| 1030_vae_mle.zip | GitHub Releases |

### 解压

```bash
# 1. 创建目录
mkdir -p data/raw/PopBuTFy data/processed/PopBuTFy_new/spk_emb data/binary/PopBuTFyENSpkEM_new
mkdir -p checkpoints/1009_pretrain_asr_english checkpoints/1012_hifigan_all_songs_nsf checkpoints/vae_mle

# 2. 数据集 (注意: zip 内含 data/ 前缀)
unzip PopBuTFy.zip -d data/raw/PopBuTFy/
unzip text_labels.zip -d data/raw/PopBuTFy/
# 合并 data/ 子目录到根目录
python restructure_raw.py

# 3. 检查点
unzip 1009_pretrain_asr_english.zip -d checkpoints/1009_pretrain_asr_english/
unzip 1012_hifigan_all_songs_nsf.zip -d checkpoints/1012_hifigan_all_songs_nsf/
unzip 1030_vae_mle.zip -d checkpoints/vae_mle/
# 1030 zip 包含多一层目录, 将文件上移一级
```

## 三、数据预处理

> **注意**: 以下所有命令在 `repo/` 目录下执行。
> 原因: YAML base_config 路径依赖 CWD 解析。

### Step 4a: 说话人嵌入提取

```bash
cd repo

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ../save_spkemb_single.py \
  --config egs/datasets/audio/PopBuTFy/save_emb.yaml \
  --hparams "processed_data_dir=../data/raw/PopBuTFy,raw_data_dir=../data/raw/PopBuTFy"
```

- 输出: `data/processed/PopBuTFy_new/spk_emb/*.npy`
- 耗时: ~40 分钟 (GPU, 28,965 句)

### Step 4b: 数据集二值化

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ../bin_para_single.py \
  --config egs/datasets/audio/PopBuTFy/para_bin.yaml \
  --hparams "processed_data_dir=../data/raw/PopBuTFy,raw_data_dir=../data/raw/PopBuTFy,spk_emb_data_dir=data/processed/PopBuTFy_new/spk_emb"
```

- 输出: `data/binary/PopBuTFyENSpkEM_new/` (索引数据集 + 统计)
- 耗时: ~30 分钟 (单进程)

### Step 4b 补充: 生成 _lengths.npy

推理数据集加载器需要 `{prefix}_lengths.npy` (mel 长度数组):

```bash
PYTHONPATH=. ../.venv/Scripts/python.exe -c "
from utils.indexed_datasets import IndexedDataset
import numpy as np
for prefix in ['valid', 'test', 'train']:
    ds = IndexedDataset('data/binary/PopBuTFyENSpkEM_new/' + prefix)
    lengths = [max(ds[i]['mel'].shape[0], ds[i].get('prof_mel', ds[i]['mel']).shape[0]) for i in range(len(ds))]
    np.save(f'data/binary/PopBuTFyENSpkEM_new/{prefix}_lengths.npy', np.array(lengths, dtype=np.int32))
"
```

## 四、推理

> 注意: 推理需 F0/pitch 字段，当前二值化未保存此字段。需补充后才能运行。

### 单进程推理

```bash
cd repo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/run.py \
  --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml \
  --exp_name vae_mle \
  --infer \
  --hparams "binary_data_dir=data/binary/PopBuTFyENSpkEM_new,\
processed_data_dir=../data/raw/PopBuTFy,\
raw_data_dir=../data/raw/PopBuTFy,\
vocoder_ckpt=../checkpoints/1012_hifigan_all_songs_nsf,\
pretrain_asr_ckpt=../checkpoints/1009_pretrain_asr_english,\
ds_workers=0"
```

输出: `checkpoints/vae_mle/generated_*/wavs/*.wav`

## 五、常见问题

### Q: Windows 报 `OSError: [WinError 1455] 页面文件太小`
**A**: Windows 多进程 CUDA DLL 加载导致页面溢出。使用单进程版本 (`ds_workers=0` / `save_spkemb_single.py` / `bin_para_single.py`)。

### Q: `librosa.filters.mel()` 报 TypeError
**A**: librosa ≥ 0.10 改用 keyword-only 参数。已将全部 3 处调用修复。

### Q: ASR 检查点加载报 `size mismatch for token_embed.weight`
**A**: phone_set.json 需恰好 78 个 token (78 + 10 保留 = 88 匹配检查点)。

### Q: `np.Inf` 报 AttributeError
**A**: 使用 `np.inf` 替代。NumPy 2.0+ 移除了 `np.Inf`。

### Q: HiFi-GAN 报 `IndexError: list index out of range`
**A**: 检查点路径反斜杠导致 regex 匹配失败。已修复为统一路径分隔符。

### Q: GPU 显存不足
**A**: 尝试减小 `max_tokens` (默认 40000)。或在推理时使用 `--hparams max_tokens=20000`。

## 六、项目文件结构 (修复后)

```
repo/
├── data_gen/
│   ├── singing/
│   │   ├── binarize.py           (未修改)
│   │   └── binarize_para.py      [已修复] 添加 _phone_encoder()
│   └── tts/
│       ├── base_binarizer.py     (未修改)
│       └── data_gen_utils.py     [已修复] mel() keyword args
├── modules/
│   └── parallel_wavegan/
│       └── stft_loss.py          [已修复] mel() keyword args
├── tasks/
│   ├── singing/
│   │   └── svb_vae_task.py      (未修改，但需关注 phone_set 维度)
│   └── tts/
│       └── dataset_utils.py      [已修复] int() 类型安全
├── utils/
│   ├── audio.py                  [已修复] mel() + np.complex
│   ├── hparams.py                (未修改，但 __init__ bug 需注意)
│   └── trainer.py                [已修复] np.Inf → np.inf
├── vocoders/
│   └── hifigan.py                [已修复] 路径分隔符
├── requirements.txt              [已废弃] 请用 UV 命令安装
└── README.md                     (未修改)
```
