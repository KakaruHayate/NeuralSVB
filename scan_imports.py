"""Scan all .py files in repo/ to find external package imports.

Usage: uv run python scan_imports.py
Output: prints a deduplicated list of import names, then a pip install command.
"""
import ast
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / 'repo'

# Known stdlib modules — we skip these
STDLIB = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'builtins',
    'codecs', 'collections', 'concurrent', 'configparser', 'copy', 'ctypes',
    'csv', 'datetime', 'decimal', 'dis', 'enum', 'fnmatch', 'functools', 'gc',
    'glob', 'gzip', 'hashlib', 'html', 'http', 'importlib', 'inspect', 'io',
    'itertools', 'json', 'keyword', 'logging', 'lzma', 'math', 'mmap',
    'multiprocessing', 'numbers', 'operator', 'os', 'pathlib', 'pickle',
    'platform', 'pprint', 'queue', 'random', 're', 'reprlib', 'select',
    'shlex', 'shutil', 'signal', 'socket', 'sqlite3', 'string', 'struct',
    'subprocess', 'sys', 'tarfile', 'tempfile', 'textwrap', 'threading',
    'time', 'tkinter', 'traceback', 'tracemalloc', 'types', 'typing',
    'unicodedata', 'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml',
    'zipfile',
}

# Local project modules (part of this repo) — skip these
LOCAL_MODULES = {
    'tasks', 'modules', 'utils', 'data_gen', 'vocoders',
    'parallel_wavegan', 'melgan',  # these are vendored under modules/
    'stft_loss', 'upsample', 'residual_block', 'causal_conv',  # local
    'radam',  # local optimizer from modules/
}

# Known third-party-to-PyPI mappings
IMPORT2PKG = {
    'yaml': 'pyyaml',
    'bs4': 'beautifulsoup4',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'tqdm': 'tqdm',
    'Cython': 'cython',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'tensorboard': 'tensorboard',
    'tensorboardX': 'tensorboardx',
    'transformers': 'transformers',
    'tokenizers': 'tokenizers',
    'pytorch_lightning': 'pytorch-lightning',
    'lightning': 'pytorch-lightning',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'torchaudio': 'torchaudio',
    'einops': 'einops',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'soxr': 'soxr',
    'resampy': 'resampy',
    'pyworld': 'pyworld',
    'pysptk': 'pysptk',
    'praat_parselmouth': 'praat-parselmouth',
    'parselmouth': 'praat-parselmouth',
    'resemblyzer': 'resemblyzer',
    'pycwt': 'pycwt',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'hparams': 'hparams',
    'g2p_en': 'g2p-en',
    'g2pM': 'g2pm',
    'pypinyin': 'pypinyin',
    'htk': 'htk-io',
    'jiwer': 'jiwer',
    'mcd': 'mcd',
    'Distance': 'distance',
    'tslearn': 'tslearn',
    'pytest': 'pytest',
    'tqdm': 'tqdm',
}


def main():
    imports = set()

    for py in sorted(REPO.rglob('*.py')):
        try:
            tree = ast.parse(py.read_text(encoding='utf-8'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    if top not in STDLIB and top not in LOCAL_MODULES:
                        imports.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split('.')[0]
                    if top not in STDLIB and top not in LOCAL_MODULES:
                        imports.add(top)

    # Try to figure out which are installed in current env
    # (empty naive check)
    print(f"=== Found {len(imports)} unique non-stdlib import names ===\n")
    for name in sorted(imports):
        pkg = IMPORT2PKG.get(name, name)
        extras = ''
        print(f"  {name:30s} → pip install {pkg}")

    # Generate pip install command
    pkgs = sorted({IMPORT2PKG.get(n, n) for n in imports})
    print(f"\n=== pip install command ===\n")
    print('uv pip install \\')
    for p in pkgs:
        print(f'  {p} \\')
    print('  # end')

    print(f"\n=== Total unique packages: {len(pkgs)} ===")


if __name__ == '__main__':
    main()
