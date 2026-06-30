# NeuralSVB — 复现代码修复清单

## 概述

NeuralSVB (ACL 2022) 开源代码存在多种兼容性问题。本项目针对：
- Python 3.12 / torch 2.x / numpy 2.x / librosa 0.11+ 版本兼容
- Windows 平台多进程 CUDA DLL 加载问题
- 数据预处理单进程适配
- config 路径解析、检查点加载等

记录所有修改以便 fork 提交。

---

## 一、依赖环境 (requirements.txt 重写)

原有 `Requirements.txt` 因版本过旧（torch 1.9.1, numpy 1.21.5）基本不可用。
通过 [scan_imports.py](scan_imports.py) AST 遍历全部 `.py` 文件提取真实依赖。

**验证通过的 27 个包** (Python 3.12 + CUDA 12.4):

| 类别 | 包 | 用途 |
|------|-----|------|
| 深度学习 | torch, torchvision, torchaudio | 核心框架 |
| 音频 | librosa, soundfile, resampy, pyworld, praat-parselmouth, pysptk | 音频 I/O, F0/Mel 提取 |
| 语音特征 | resemblyzer | 说话人嵌入 |
| NLP | transformers, tokenizers, g2p-en, g2pM, pypinyin, jieba, nltk | ASR 文本处理 |
| 科学计算 | numpy, scipy, scikit-learn, scikit-image | 数据处理 |
| 工具 | pyyaml, h5py, tqdm, einops, tensorboardX, matplotlib, pandas | I/O/Logging/Plot |
| 信号处理 | pycwt, webrtcvad, pyloudnorm | 小波变换/VAD/响度 |
| 序列模型 | tslearn, torchcrf | DTW/CRF |

**安装方式**:
```bash
uv venv --python 3.12 .venv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install numpy scipy scikit-learn einops pyyaml tqdm h5py librosa soundfile resampy pyworld pysptk praat-parselmouth resemblyzer pycwt transformers tokenizers tensorboardX matplotlib pandas packaging chardet six jieba nltk g2p-en pypinyin webrtcvad pyloudnorm Cython scikit-image g2pm
```

---

## 二、代码 Bug 修复

### 2.1 librosa.filters.mel() API 兼容性

**文件**: `data_gen/tts/data_gen_utils.py`, `utils/audio.py`, `modules/parallel_wavegan/stft_loss.py`

**问题**: librosa 0.10+ 将 `mel()` 函数参数改为 keyword-only。旧代码使用位置参数报 `TypeError: mel() takes 0 positional arguments but 5 were given`。

**修复**: 改为 keyword args:

```python
# 旧 (librosa < 0.10):
mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)

# 新 (librosa >= 0.10):
mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
```

涉及 3 处调用。

### 2.2 NumPy 2.x 废弃别名

**文件**: `utils/trainer.py`

```python
# 旧:
self.best_val_results = np.Inf if monitor_mode == 'min' else -np.Inf
# 新:
self.best_val_results = np.inf if monitor_mode == 'min' else -np.inf
```

**文件**: `utils/audio.py`

```python
# 旧:
S_complex = np.abs(S).astype(np.complex)
# 新:
S_complex = np.abs(S).astype(complex)
```

### 2.3 F0 提取帧数不匹配 (新版本库兼容)

**文件**: `data_gen/tts/data_gen_utils.py`

**问题**: 不同库版本（librosa, parselmouth）计算的帧数差异导致 `rpad` 为负数，

```python
# 旧: rpad 可能为负，np.pad 无法处理
rpad = len(mel) - len(f0) - lpad
f0 = np.pad(f0, [[lpad, rpad]], mode='constant')  # ValueError

# 新: 先截断 F0 再 padding
if rpad < 0:
    f0 = f0[:len(mel) - lpad]
    rpad = len(mel) - len(f0) - lpad
f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
```

**文件**: `vocoders/hifigan.py`

**问题**: Windows 路径使用反斜杠 `\`，但 regex 用 `/` 分隔符匹配，导致 `re.findall` 返回空列表引发 `IndexError`。

```python
# 旧: regex 包含 base_dir 路径
lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\\d+).ckpt', x)[0])

# 新: 仅匹配文件名，且统一路径分隔符
lambda x: -int(re.findall(f'model_ckpt_steps_(\\d+).ckpt', x.replace('\\\\', '/'))[0])
```

### 2.4 数据集 PhoneSet 维度不匹配

**文件**: `tasks/singing/svb_vae_task.py` (未修改文件，需关注)

**问题**: 代码从 `phone_set.json` 读取 token 列表，传递给 `MleSVBVAE(len(phone_list) + 10)`。预训练 ASR 检查点 `token_embed.weight` 形状为 `[88, 256]`，因此 `phone_list` 需恰好包含 **78** 个 token（78 + 10 = 88）。

**定位**: 模型构造时需保证 `len(phone_list) + 10 == asr_ckpt_dict_size`。

### 2.5 数据集加载器 int32 类型问题

**文件**: `tasks/tts/dataset_utils.py`

**问题**: `_lengths.npy` 中的 `np.int32` 值与 Python `int` 的相等比较在新版 numpy 中行为不同。

```python
# 旧:
assert max(len(item['mel']), len(item['prof_mel'])) == self.sizes[index]
# 新:
assert max(len(item['mel']), len(item['prof_mel'])) == int(self.sizes[index])
```

---

## 三、Windows 平台适配

### 3.1 多进程 CUDA DLL 溢出

**问题**: Windows 的 `multiprocessing.spawn` 会在子进程中重载 CUDA DLL，导致页面文件不足 (`OSError: [WinError 1455]`)。原始代码使用 `chunked_multiprocess_run` 进行多进程数据预处理。

**解决方案**: 创建单进程版本的预处理脚本:
- `save_spkemb_single.py` — 说话人嵌入提取（替代 `save_emb.yaml` 流程）
- `bin_para_single.py` — 数据集二值化（替代 `para_bin.yaml` 流程）

### 3.2 `_phone_encoder()` 缺失

**文件**: `data_gen/singing/binarize_para.py`

**问题**: `PopBuTFyENSpkEMBinarizer` 未实现 `_phone_encoder()` 方法，导致 `builder` 操作前抛出 `NotImplementedError`。

**修复**: 添加 `_phone_encoder()` 方法，从文本标签构建 phone_set 并保存到 `phone_set.json`。

---

## 四、数据预处理流程标准化

### 4.1 目录结构

原始数据集 zip 包存在目录前缀不一致：
- `PopBuTFy.zip`: `<base>/data/*/*.mp3` (带 `data/` 前缀)
- `text_labels.zip`: `<base>/*/*.txt` (无前缀)
- `PopBuTFy-preview.zip`: `<base>/*/*.mp3` (无前缀)

**处理**: 解压后统一为 `data/raw/PopBuTFy/data/*/*.{mp3,txt}`。

### 4.2 路径解析问题

**问题**: YAML 配置链中的 `base_config` 路径不带 `./` 前缀时从 CWD 解析，带 `./` 从 YAML 所在目录解析。运行脚本时工作目录必须为 `repo/`。

**经验**: 所有命令需从 `repo/` 目录执行，`processed_data_dir` / `raw_data_dir` 等使用 `../data/raw/PopBuTFy` 相对路径。

---

## 五、推理 Pipeline 当前状态

### 已完成
- [x] 依赖安装（27 个包验证通过）
- [x] 说话人嵌入提取 (28,965 句)
- [x] 数据集二值化 (12,636 对业余-专业)
- [x] 检查点加载通过 (ASR / HiFi-GAN / VAE)
- [x] 单进程推理模式 (`ds_workers=0`)

### 进行中
- [ ] F0/pitch 特征补全（二值化未保存此字段）

### 待解决
- [ ] VAE 模型前向推理输出 WAV 文件（需要 F0 就位后测试）
- [ ] 验证 WAV 输出音质

---

## 六、文件清单

### 新增脚本
| 文件 | 用途 |
|------|------|
| `scan_imports.py` | AST 遍历提取真实依赖 |
| `save_spkemb_single.py` | 单进程说话人嵌入提取 |
| `bin_para_single.py` | 单进程数据集二值化 |
| `merge_data_dir.py` | 数据目录结构合并 |
| `restructure_raw.py` | 目录重组 |
| `debug_bin_item.py` | 二值化调试 |

### 修改原仓库文件
| 文件 | 修改内容 |
|------|----------|
| `data_gen/tts/data_gen_utils.py` | line 130: `mel()` keyword args |
| `utils/audio.py` | line 100: `mel()` keyword args; line 37: `np.complex`→`complex` |
| `modules/parallel_wavegan/stft_loss.py` | line 45: `mel()` keyword args |
| `utils/trainer.py` | line 71: `np.Inf`→`np.inf` |
| `vocoders/hifigan.py` | line 45-46: regex 路径分隔符修正 |
| `data_gen/singing/binarize_para.py` | line 219+: 添加 `_phone_encoder()` |
| `tasks/tts/dataset_utils.py` | line 59: `int()` cast 类型安全 |
| `data_gen/tts/data_gen_utils.py` | line 174+: F0 帧数兼容; line 329: `np.int`→`int` |
| `utils/pitch_utils.py` | line 144: `np.int`→`int` |