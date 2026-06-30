import glob
import json
import os
import re

import librosa
import torch

import utils
from modules.hifigan.hifigan import HifiGanGenerator
from utils.hparams import hparams, set_hparams
from vocoders.base_vocoder import register_vocoder
from vocoders.pwg import PWG
from vocoders.vocoder_utils import denoise


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if '.yaml' in config_path:
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
    elif '.json' in config_path:
        config = json.load(open(config_path, 'r'))
        state = ckpt_dict["generator"]

    model = HifiGanGenerator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class HifiGAN(PWG):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        if os.path.exists(config_path):
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: -int(re.findall(f'model_ckpt_steps_(\d+).ckpt', x.replace('\\', '/'))[0]))[0]
            print('| load HifiGAN: ', ckpt)
            self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
            # Load mel conversion matrix (VAE fmin=50→HiFi fmin=0) if available
            self.mel_convert = None
            conv_path = os.path.join(os.path.dirname(base_dir), 'mel_vae2hifi.npy')
            if os.path.exists(conv_path):
                import numpy as np
                self.mel_convert = torch.from_numpy(np.load(conv_path)).float()
                print(f'| Loaded mel conversion matrix from {conv_path}')
        else:
            config_path = f'{base_dir}/config.json'
            ckpt = f'{base_dir}/generator_v1'
            if os.path.exists(config_path):
                self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)  # [1, 80, T]
            # Apply mel conversion matrix if available (VAE→HiFi mel basis shift)
            if self.mel_convert is not None:
                c = (self.mel_convert.to(device) @ c.squeeze(0)).unsqueeze(0)  # [1, 80, T]
            with utils.Timer('hifigan', enable=hparams['profile_infer']):
                f0 = kwargs.get('f0')
                if f0 is not None:
                    f0 = torch.FloatTensor(f0[None, :]).to(device)
                    y = self.model(c, f0).view(-1)
                else:
                    y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out

    # @staticmethod
    # def wav2spec(wav_fn, **kwargs):
    #     wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
    #     wav_torch = torch.FloatTensor(wav)[None, :]
    #     mel = mel_spectrogram(wav_torch, hparams).numpy()[0]
    #     return wav, mel.T
