#!/usr/bin/env python3
"""Debug: single item processing in binarization."""
import sys, os, json, numpy as np, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'repo'))

# Simulate command-line args for set_hparams
if len(sys.argv) < 2:
    sys.argv = ['', '--config', 'egs/datasets/audio/PopBuTFy/para_bin.yaml',
                '--hparams', 'processed_data_dir=../data/raw/PopBuTFy,raw_data_dir=../data/raw/PopBuTFy,spk_emb_data_dir=data/processed/PopBuTFy_new/spk_emb']

from utils.hparams import set_hparams, hparams
h = set_hparams()
print(f"CWD: {os.getcwd()}")
print(f"raw_data_dir: {h.get('raw_data_dir')}")
print(f"processed_data_dir: {h.get('processed_data_dir')}")
print(f"binary_data_dir: {h.get('binary_data_dir')}")
print(f"spk_emb_data_dir: {h.get('spk_emb_data_dir')}")

# Check actual paths
raw_dir = h['raw_data_dir'] if os.path.isabs(h['raw_data_dir']) else os.path.join(os.getcwd(), h['raw_data_dir'])
print(f"Absolute raw dir: {os.path.abspath(raw_dir)}")
print(f"Exists: {os.path.exists(os.path.abspath(raw_dir))}")

from data_gen.singing.binarize_para import PopBuTFyENSpkEMBinarizer

b = PopBuTFyENSpkEMBinarizer()
b.load_meta_data()
print(f"Found {len(b.item_names)} items")
print(f"#singing items: {sum(1 for n in b.item_names if '#singing#' in n)}")

# Build spk map
b.spk_map = b.build_spk_map()
print(f"spk_map: {b.spk_map}")

# First amateur-professional pair
for item_name in b.item_names:
    if '#singing#' not in item_name or 'Professional' in item_name:
        continue
    prof_item = item_name.replace('Amateur', 'Professional')
    wav = b.item2wavfn.get(item_name)
    pwav = b.item2wavfn.get(prof_item)
    if wav is None or pwav is None:
        continue
    if not os.path.exists(wav) or not os.path.exists(pwav):
        print(f"  MISSING FILE: wav={wav}, pwav={pwav}")
        continue
    print(f"\nTest item: {item_name}")
    print(f"  wav: {wav}")
    print(f"  prof: {pwav}")
    txt_fn = wav.replace('.mp3', '.txt')
    if os.path.exists(txt_fn):
        print(f"  txt: {open(txt_fn).readlines()[0].strip()}")
    else:
        print(f"  NO TXT: {txt_fn}")

    # Try wav2spec
    try:
        from vocoders.base_vocoder import get_vocoder_cls
        vocoder_cls = get_vocoder_cls(hparams)
        # wav2spec internally calls process_utterance which uses librosa.filters.mel
        wav_data, mel = vocoder_cls.wav2spec(wav)
        profwav_data, profmel = vocoder_cls.wav2spec(pwav)
        print(f"  wav2spec OK: mel {mel.shape}, profmel {profmel.shape}")
        print(f"  wav: {wav_data.shape}, profwav: {profwav_data.shape}")
        break
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        break
