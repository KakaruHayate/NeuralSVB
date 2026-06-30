#!/usr/bin/env python3
"""Single-process version of SaveSpkEmb binarizer.

Windows multiprocessing spawn doesn't inherit global hparams/CUDA context,
so we run everything in the main process.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'repo'))
os.environ["OMP_NUM_THREADS"] = "1"

from utils.hparams import set_hparams, hparams
from data_gen.singing.binarize_para import SaveSpkEmb
from tqdm import tqdm
from resemblyzer import VoiceEncoder
import numpy as np
import traceback

def binarize_single_process():
    # Load config
    set_hparams()

    # Instantiate binarizer
    b = SaveSpkEmb()
    b.load_meta_data()

    # Build output dirs
    os.makedirs(hparams['binary_data_dir'], exist_ok=True)
    b.spk_map = b.build_spk_map()
    print("| spk_map: ", b.spk_map)
    import json
    spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
    json.dump(b.spk_map, open(spk_map_fn, 'w'))

    # Process each split
    voice_encoder = VoiceEncoder().cuda()

    for prefix in ['valid', 'test', 'train']:
        if prefix == 'valid':
            item_names = b.valid_item_names
        elif prefix == 'test':
            item_names = b.test_item_names
        else:
            item_names = b.train_item_names

        print(f"\nProcessing {prefix}: {len(item_names)} items")
        spk_emb_dir = hparams['spk_emb_data_dir']
        os.makedirs(spk_emb_dir, exist_ok=True)

        for item_name in tqdm(item_names, desc=f'spk_emb_{prefix}'):
            wav_fn = b.item2wavfn.get(item_name)
            if wav_fn is None or not os.path.exists(wav_fn):
                continue
            spk_id = b.item_name2spk_id(item_name)
            try:
                # process_item
                from vocoders.base_vocoder import get_vocoder_cls
                wav, mel = get_vocoder_cls(hparams).wav2spec(wav_fn)
                item = {
                    'item_name': item_name,
                    'wav_fn': wav_fn,
                    'spk_id': spk_id,
                    'mel': mel,
                    'wav': wav,
                    'sec': len(wav) / hparams['audio_sample_rate'],
                    'len': mel.shape[0],
                }
                item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])
                np.save(os.path.join(spk_emb_dir, item['item_name'] + '.npy'), item['spk_embed'])
            except Exception as e:
                print(f"Error processing {item_name}: {e}")
                traceback.print_exc()

    print("\nDone! Speaker embeddings saved to:", spk_emb_dir)

if __name__ == '__main__':
    binarize_single_process()
