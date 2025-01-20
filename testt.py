import os
import sys
import shutil
from pathlib import Path
from paket.modules import VC
from paket.uvr_modules import uvr
from paket.config import Config
from sklearn.cluster import MiniBatchKMeans
from dotenv import load_dotenv
import torch
import numpy as np
import faiss
import pathlib
import json
from subprocess import Popen

load_dotenv()

BASE_DIR = Path.cwd()
now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
weight_uvr5 = ('model')

uvr5_names = []
for name in os.listdir(weight_uvr5):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

def main():
    config = Config()  # Inisialisasi Config di dalam main()
    vc = VC(config)

    if config.dml == True:

        def forward_dml(ctx, x, scale):
            ctx.scale = scale
            res = x.clone().detach()
            return res

    input_path = config.input_path  # Ambil dari config
    opt_root = config.output_folder  # Ambil dari config
    model_choose = config.model  # Ambil dari config
    agg = config.agg  # Ambil dari config
    format0 = config.format  # Ambil dari config
    opt_vocal_root = os.path.join(opt_root, "vocals")
    opt_ins_root = os.path.join(opt_root, "instruments")

    os.makedirs(opt_vocal_root, exist_ok=True)
    os.makedirs(opt_ins_root, exist_ok=True)

    # Cek apakah input_path adalah file atau folder
    if os.path.isfile(input_path):
        # Proses file tunggal
        print(f"Processing single file: {input_path}")
        for result in uvr(
            model_choose,
            None,  # dir_wav_input di-set None karena kita memproses file tunggal
            opt_vocal_root,
            input_path,  # Masukkan path file audio di sini
            opt_ins_root,
            agg,
            format0,
        ):
            print(result)
    elif os.path.isdir(input_path):
        # Proses batch dalam folder
        print(f"Processing batch files in folder: {input_path}")
        for result in uvr(
            model_choose,
            input_path,  # Masukkan path folder di sini
            opt_vocal_root,
            None,  # wav_inputs di-set None karena kita memproses folder
            opt_ins_root,
            agg,
            format0,
        ):
            print(result)
    else:
        print("Invalid input path. Please provide a valid file or folder path.")

if __name__ == "__main__":
    main()
