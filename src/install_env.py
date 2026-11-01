"""
Install the exact environment the experiments were developed in (Colab, T4, Python 3.13).

Why this is needed: Colab's default torch (2.11 + CUDA 13) has NO prebuilt mamba_ssm / causal_conv1d wheels, and
compiling them from source crashes the Colab session (out of memory). The prebuilt wheels exist for torch 2.10 + cu12
(x86_64), so we install that torch and the matching wheels. We also remove torchvision/torchaudio, whose CUDA-13 builds
no longer match and make `import transformers` crash.

Run once per fresh runtime, then Runtime -> Restart session. Safe to run from a notebook cell:  %run install_env.py
"""
import json
import subprocess
import sys
import urllib.request


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("$", cmd[:150])
    print(r.stdout[-700:])
    print(r.stderr[-700:])
    return r.returncode


if sh("pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128") != 0:
    sh("pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126")
sh("pip uninstall -y torchvision torchaudio")
sh("pip install -q einops packaging pandas huggingface_hub")

py = f"cp{sys.version_info.major}{sys.version_info.minor}"


def find(repo):
    for page in (1, 2, 3, 4, 5):
        url = f"https://api.github.com/repos/{repo}/releases?per_page=50&page={page}"
        for r in json.load(urllib.request.urlopen(url)):
            for a in r["assets"]:
                n = a["name"]
                if n.endswith(".whl") and f"cu12torch2.10cxx11abiTRUE-{py}-{py}-linux_x86_64" in n:
                    return n, a["browser_download_url"]
    return None, None


for repo in ["Dao-AILab/causal-conv1d", "state-spaces/mamba"]:
    n, u = find(repo)
    print(repo, "->", n)
    if u:
        sh(f"pip install --no-deps '{u}'")
    else:
        print("NO x86_64 WHEEL FOUND for", py, "- tell the assistant; a different torch pin is needed")

sh("python -c \"import mamba_ssm, causal_conv1d; print('IMPORT OK', mamba_ssm.__version__, causal_conv1d.__version__)\"")
print("\nNOW: Runtime -> Restart session (the installed wheels survive the restart).")
