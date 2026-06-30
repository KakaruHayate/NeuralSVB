# Learning the Beauty in Songs: Neural Singing Voice Beautifier
---
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/NeuralSVB)](https://github.com/MoonInTheRiver/NeuralSVB)
![visitors](https://visitor-badge.glitch.me/badge?page_id=moonintheriver/NeuralSVB)

<div align="center">
    <a href="https://neuralsvb.github.io" target="_blank">Demo&nbsp;Page</a>
</div>


This repository is the official PyTorch implementation of our ACL-2022 [paper](https://arxiv.org/abs/2202.13277). 


## 0. Dataset (PopBuTFy) Acquirement
### Audio samples
- You can download the dataset from [here](https://drive.google.com/file/d/1IKFp7y1WeYGrwXgJ0HC3rdPj54WoqIsU/view?usp=sharing). Please send us an email for registration (See in [apply_form](resources/apply_form.md)).
- Dataset [preview](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/PopBuTFy-preview.zip).

### Text labels
NeuralSVB does not need text as input, but the ASR model to extract PPG needs text. Thus we also provide the [text labels](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/text_labels.zip) of PopBuTFy. 
<!-- We recommend mixing [LibriTTS](https://www.openslr.org/60/) with PopBuTFy to train the ASR model. -->

## 1. Preparation

### Environment Preparation
Most of the required packages are in https://github.com/NATSpeech/NATSpeech/blob/main/requirements.txt

Or you can prepare environments with the Requirements.txt file in the repository directory.
```sh
pip install Requirements.txt
```
### Data Preparation


1. Extract embeddings of vocal timbre:
    ```sh 
    CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config egs/datasets/audio/PopBuTFy/save_emb.yaml
    ```
2. Pack the dataset:
    ```sh 
    CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config egs/datasets/audio/PopBuTFy/para_bin.yaml
    ```


### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/1012_hifigan_all_songs_nsf.zip) which is specially designed for SVS with NSF mechanism.

Please unzip pre-trained vocoder into `checkpoints` before training your acoustic model.

This singing vocoder is trained on 100+ hours singing data (including Chinese and English songs). 

### PPG Extractor Preparation
We provide the pre-trained model of [PPG Extractor](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/1009_pretrain_asr_english.zip).

Please unzip pre-trained PPG extractor into `checkpoints` before training your acoustic model.


After the instructions above, the directory structure should be as follows:

```
.
|--data
    |--processed
        |--PopBuTFy (unzip PopBuTFy.zip)
            |--data
                |--directories containing wavs
    |--binary
        |--PopBuTFyENSpkEM
|--checkpoints
    |--1009_pretrain_asr_english
        |--
        |--config.yaml
    |--1012_hifigan_all_songs_nsf
        |--
        |--config.yaml
```


## 2. Training Example

```sh
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml --exp_name exp_name --reset
```

## 3. Inference
### Inference from packed test set

```sh
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml --exp_name exp_name --reset --infer
```
Inference results will be saved in `./checkpoints/EXP_NAME/generated_` by default.

We provided:
 - the [pre-trained model](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/1030_vae_mle.zip) of NSVB (en version);

Remember to put the pre-trained models in `checkpoints` directory.

### Inference from raw inputs
WIP.

---

## 环境搭建与代码修复（中文）

> 针对 Python 3.12 / torch 2.x / numpy 2.x / librosa 0.11+ / Windows 的兼容性修复记录。

### 环境要求

- Python ≥ 3.10（推荐 3.12）
- NVIDIA GPU + CUDA（CPU 推理极慢）
- 磁盘空间 ≥ 15GB

### 快速开始

```bash
# 1. 创建虚拟环境（推荐 UV）
uv venv --python 3.12 .venv

# 2. 安装 PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装其余依赖
uv pip install numpy scipy scikit-learn einops pyyaml tqdm h5py \
  librosa soundfile resampy pyworld pysptk praat-parselmouth \
  resemblyzer pycwt transformers tokenizers tensorboardX matplotlib \
  pandas packaging chardet six jieba nltk g2p-en pypinyin \
  webrtcvad pyloudnorm Cython scikit-image g2pm tslearn
```

### 数据预处理（Windows 用户注意）

原始代码使用多进程并行处理数据，Windows 下 `multiprocessing.spawn` 会导致 CUDA DLL 加载失败（页面文件溢出）。
提供单进程替代脚本：

```bash
# Step 1: 提取说话人嵌入（替代 save_emb.yaml）
cd repo
PYTHONPATH=. python ../save_spkemb_single.py \
  --config egs/datasets/audio/PopBuTFy/save_emb.yaml \
  --hparams "processed_data_dir=../data/raw/PopBuTFy,raw_data_dir=../data/raw/PopBuTFy"

# Step 2: 数据集二值化（替代 para_bin.yaml）
PYTHONPATH=. python ../bin_para_single.py \
  --config egs/datasets/audio/PopBuTFy/para_bin.yaml \
  --hparams "processed_data_dir=../data/raw/PopBuTFy,raw_data_dir=../data/raw/PopBuTFy,\
spk_emb_data_dir=data/processed/PopBuTFy_new/spk_emb"
```

### 推理

```bash
cd repo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/run.py \
  --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml \
  --exp_name vae_mle --infer \
  --hparams "binary_data_dir=data/binary/PopBuTFyENSpkEM_new,\
processed_data_dir=../data/raw/PopBuTFy,\
raw_data_dir=../data/raw/PopBuTFy,\
vocoder_ckpt=../checkpoints/1012_hifigan_all_songs_nsf,\
pretrain_asr_ckpt=../checkpoints/1009_pretrain_asr_english,\
ds_workers=0,normalize_pitch=False,\
f0_mean=268.2,f0_std=87.5"
```

### 已知问题与修复

| # | 问题 | 修复 | 涉及文件 |
|---|------|------|----------|
| 1 | `librosa.filters.mel()` 在新版中改为 keyword-only 参数 | 改为 keyword args | `data_gen_utils.py`, `audio.py`, `stft_loss.py` |
| 2 | `np.Inf`、`np.complex`、`np.int` 在 NumPy 2.0+ 被移除 | 替换为 `np.inf`、`complex`、`int` | `trainer.py`, `audio.py`, `pitch_utils.py` |
| 3 | Windows 路径反斜杠导致 HiFi-GAN 检查点加载失败 | 统一路径分隔符 | `hifigan.py` |
| 4 | Windows 多进程 CUDA DLL 溢出 | 单进程替代脚本 | `save_spkemb_single.py`, `bin_para_single.py` |
| 5 | 数据集 phone_set 维度与 ASR 检查点不匹配 | phone_set 78 项 + 10 保留 = 88 | `phone_set.json` |
| 6 | `_phone_encoder()` 未实现导致二值化失败 | 补充实现 | `binarize_para.py` |
| 7 | F0 提取帧数不匹配（新版库兼容） | 先截断再 padding | `data_gen_utils.py` |
| 8 | `saving_result_pool` 多进程写入时 hparams 丢失 | Windows 下替换为 FakePool | `tts.py`, `fs2.py` |
| 9 | 推理时 `f0_mean/f0_std` 缺失 | 从数据计算 F0 统计，运行时传入 | `pitch_utils.py`, `hparams.py` |
| 10 | Windows 文件名含 `?` 等非法字符 | 过滤替换 | `svb_vae_task.py` |

### 音频质量问题（结题）

推理输出的 a2p（业余→专业）mel 范围约为 [-0.6, 0.7]，而正常 log-mel 范围为 [-6, -0.4]。
语义内容存在但 HiFi-GAN 合成后为爆音。

**完整排查结论：**

| 假设 | 验证结果 |
|------|----------|
| librosa API 导致 mel 计算错误 | ❌ 已修复 keyword args |
| F0 归一化缺失（f0_mean/f0_std = None） | ❌ 从数据统计并传入后无改善 |
| HiFi-GAN 声码器本身问题 | ❌ GT mel 过声码器输出正常（mean≈0, std≈0.03） |
| VAE→HiFi mel 基不匹配（fmin=50 vs 0） | ❌ 重采样 + 分布纠正后 VAE 输出仍过于平滑（std=0.012 vs GT 0.029） |
| Git 历史中 MrZixi 的 param scales 改动 | ❌ 全部是 NaN/Inf debug + epsilon 微调，无行为级改动 |
| VAE 检查点训练数据不匹配 | ✅ `checkpoints/vae_mle/config.yaml` 中 `binary_data_dir: molar_long_english_new`，**非 PopBuTFy** |

**根因：`1030_vae_mle` 预训练权重对应的训练数据是 Molar 数据集**（内部名 `molar_long_english_new`），而非 PopBuTFy。VAE 解码器在新数据上出现后验坍缩（posterior collapse），输出的 mel 方差极小（std ≈ 0.19 vs 目标 1.02），导致 HiFi-GAN 合成后音量偏低且含爆音。

**解决方案：**
1. 在 PopBuTFy 上重新训练 VAE（需要原训练流程）
2. 或获取与 PopBuTFy 数据分布匹配的 VAE 权重
3. 或在 Molar 数据集上运行推理

### 项目脚本说明

| 脚本 | 用途 |
|------|------|
| `scan_imports.py` | AST 遍历全部 .py 文件提取 import，生成真实依赖清单 |
| `save_spkemb_single.py` | 单进程版说话人嵌入提取（解决 Windows 多进程问题） |
| `bin_para_single.py` | 单进程版数据集二值化（含 F0/对齐/多说话人嵌入） |
| `merge_data_dir.py` | 修复 PopBuTFy.zip（带 data/ 前缀）与 text_labels.zip 的目录合并 |
| `PATCH_SUMMARY.md` | 完整修改清单 |
| `SETUP.md` | 详细环境搭建与运行指南 |

---

## Limitations
See Appendix D "Limitations and Solutions" in our [paper](https://aclanthology.org/2022.acl-long.549.pdf).

## Citation
If this repository helps your research, please cite:

    @inproceedings{liu-etal-2022-learning-beauty,
    title = "Learning the Beauty in Songs: Neural Singing Voice Beautifier",
    author = "Liu, Jinglin  and
      Li, Chengxi  and
      Ren, Yi  and
      Zhu, Zhiying  and
      Zhao, Zhou",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.549",
    pages = "7970--7983",}


## Issues
 - Before raising a issue, please check our Readme and other issues for possible solutions.
 - We will try to handle your problem in time but we could not guarantee a satisfying solution.
 - Please be friendly.

## Acknowledgements
* r9y9's [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
* Po-Hsun-Su's [ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* descriptinc's [melgan](https://github.com/descriptinc/melgan-neurips)
* Official [espnet](https://github.com/espnet/espnet)
* Official [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

The framework of this repository is based on [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger), 
and is a predecessor of [NATSpeech](https://github.com/NATSpeech/NATSpeech/). 
