#!/usr/bin/env python3
"""Single-process version of PopBuTFyENSpkEMBinarizer binarization.

Windows multiprocessing spawn doesn't inherit global hparams/CUDA context,
so we run everything in the main process.
"""
import sys, os, json, traceback, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'repo'))
os.environ["OMP_NUM_THREADS"] = "1"

from utils.hparams import set_hparams, hparams
from data_gen.singing.binarize_para import PopBuTFyENSpkEMBinarizer
from data_gen.singing.binarize import split_train_test_set
from vocoders.base_vocoder import get_vocoder_cls
from data_gen.tts.data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder, is_sil_phoneme
from data_gen.tts.base_binarizer import BinarizationError
from utils.indexed_datasets import IndexedDatasetBuilder
from resemblyzer import VoiceEncoder
from tqdm import tqdm

import random
random.seed(1234)


class SingleProcessPopBuTFyBinarizer(PopBuTFyENSpkEMBinarizer):
    """Override process_data to run single-process (no multiprocessing)."""

    @property
    def num_workers(self):
        return 1  # disable multiprocessing

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        os.makedirs(data_dir, exist_ok=True)
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        ph_lengths = []
        mel_lengths = []
        f0s = []
        total_sec = 0

        voice_encoder = VoiceEncoder().cuda() if self.binarization_args.get('with_spk_embed') else None
        spk_emb_dir = hparams.get('spk_emb_data_dir', None)
        phone_encoder = self._phone_encoder()

        meta_data = list(self.meta_data(prefix))
        print(f"Processing {prefix}: {len(meta_data)} items")

        for entry in tqdm(meta_data, desc=f'bin_{prefix}'):
            # meta_data yields 5-tuple: item_name, wav_fn, spk_id, profwavfn, item_names
            item_name, wav_fn, spk_id, profwavfn = entry[0], entry[1], entry[2], entry[3]
            try:
                item = self._process_one(item_name, wav_fn, spk_id, profwavfn,
                                         phone_encoder, voice_encoder, spk_emb_dir)
                if item is None:
                    continue
                builder.add_item(item)
                ph_lengths.append(item['len'])
                mel_lengths.append(item['mel'].shape[0])
                total_sec += item['sec']
                if 'f0' in item:
                    f0s.append(item['f0'])
            except Exception as e:
                print(f"Error processing {item_name}: {e}")
                traceback.print_exc()

        builder.finalize()
        print(f"| {prefix} set: {len(ph_lengths)} items, total {total_sec:.1f}s")

        # Save stats (matches original binarizer format)
        stats = {
            'ph_lengths': ph_lengths,
            'mel_lengths': mel_lengths,
            'f0s': f0s,
            'total_sec': total_sec,
        }
        np.save(f'{data_dir}/{prefix}_stats.npy', stats)
        # Dataset loader needs _lengths.npy (mel lengths)
        np.save(f'{data_dir}/{prefix}_lengths.npy', np.array(mel_lengths, dtype=np.int32))

    def _process_one(self, item_name, wav_fn, spk_id, profwavfn,
                     phone_encoder, voice_encoder, spk_emb_dir):
        binarization_args = self.binarization_args
        wav, mel = get_vocoder_cls(hparams).wav2spec(wav_fn)

        # Basic item
        txt_fn = wav_fn.replace('.mp3', '.txt')
        txt = open(txt_fn).readlines()[0].strip() if os.path.exists(txt_fn) else ''

        item = {
            'item_name': item_name,
            'txt': txt,
            'spk_id': spk_id,
            'mel': mel,
            'sec': len(wav) / hparams['audio_sample_rate'],
            'len': mel.shape[0],
        }

        # Professional
        profwav, profmel = get_vocoder_cls(hparams).wav2spec(profwavfn)
        item.update({
            'prof_mel': profmel,
            'prof_sec': len(profwav) / hparams['audio_sample_rate'],
            'prof_len': profmel.shape[0],
        })

        # F0 / pitch (mandatory for inference)
        if binarization_args.get('with_f0'):
            try:
                f0, pitch_coarse = get_pitch(wav, mel, hparams)
                item.update({'f0': f0.astype(np.float32), 'pitch': pitch_coarse})
            except Exception as e:
                print(f"  F0 fail {item_name}: {e}")
                item['f0'] = np.zeros(mel.shape[0], dtype=np.float32)
                item['pitch'] = np.zeros(mel.shape[0], dtype=np.int64)

            try:
                prof_f0, prof_pitch = get_pitch(profwav, profmel, hparams)
                item.update({'prof_f0': prof_f0.astype(np.float32),
                             'prof_pitch': prof_pitch})
            except Exception as e:
                print(f"  F0(prof) fail {item_name}: {e}")
                item['prof_f0'] = np.zeros(profmel.shape[0], dtype=np.float32)
                item['prof_pitch'] = np.zeros(profmel.shape[0], dtype=np.int64)

        # F0 alignment (linear interpolation as default DTW)
        f0_am = item.get('f0', np.zeros(mel.shape[0]))
        f0_pr = item.get('prof_f0', np.zeros(profmel.shape[0]))
        if len(f0_pr) > 0 and len(f0_am) > 0:
            # Simple linear alignment: stretch amateur indices to prof length
            align = np.clip(np.round(
                np.arange(len(f0_pr), dtype=np.float32) * len(f0_am) / max(len(f0_pr), 1)
            ).astype(np.int64), 0, max(len(f0_am) - 1, 0))
            item['a2p_f0_alignment'] = align

        # Speaker embedding from file
        if spk_emb_dir:
            emb_fn = os.path.join(spk_emb_dir, item_name + '.npy')
            if os.path.exists(emb_fn):
                item['spk_embed'] = np.load(emb_fn)

        # Multi-speaker embedding (use self + one random other)
        if spk_emb_dir:
            other_names = [n for n in self.item_names if n != item_name]
            if other_names:
                import random
                other = random.choice(other_names)
                other_fn = os.path.join(spk_emb_dir, other + '.npy')
            else:
                other_fn = emb_fn
            default_emb = np.zeros(256, dtype=np.float32)
            spk_a = np.load(emb_fn) if os.path.exists(emb_fn) else default_emb
            spk_b = np.load(other_fn) if os.path.exists(other_fn) else default_emb
            item['multi_spk_emb'] = np.stack([spk_a, spk_b], axis=0)

        # Remove raw wav to save space
        item.pop('wav', None)
        item.pop('prof_wav', None)

        return item


def main():
    set_hparams()

    # Instantiate binarizer
    b = SingleProcessPopBuTFyBinarizer()
    b.load_meta_data()

    # Build output dirs
    os.makedirs(hparams['binary_data_dir'], exist_ok=True)

    # Build phone encoder
    phone_encoder = b._phone_encoder()
    print(f"| Phone encoder: {len(phone_encoder)} tokens")

    # Build spk map
    b.spk_map = b.build_spk_map()
    with open(f"{hparams['binary_data_dir']}/spk_map.json", 'w') as f:
        json.dump(b.spk_map, f)
    print(f"| spk_map: {b.spk_map}")

    # Process each split
    b.process_data('valid')
    b.process_data('test')
    b.process_data('train')

    print("\nDone! Binary data saved to:", hparams['binary_data_dir'])


if __name__ == '__main__':
    main()
