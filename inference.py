import os
import subprocess
import torch
import logging
import argparse
from audio_separator.separator import Separator

# Model dan perangkat
device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

# Direktori output dan model
models_dir = "./models"
out_dir = "./outputs"

# Model yang akan digunakan
roformer_models = {
    'MelBand Roformer Kim | FT by unwa': 'mel_band_roformer_kim_ft_unwa.ckpt',
    'MelBand Roformer | De-Reverb Less Aggressive by anvuew': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
}

def pad_audio_if_needed(input_file, temp_file, min_duration=10):
    """Menambahkan silence jika audio kurang dari durasi minimum."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_file
    ], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    if duration < min_duration:
        pad_duration = min_duration - duration
        subprocess.run([
            "ffmpeg", "-i", input_file, "-af", f"apad=pad_dur={pad_duration}",
            "-c:a", "pcm_s16le", temp_file
        ])
        return temp_file
    return input_file

def perform_separation(input_file, model_name, output_folder, output_format, single_stem=None, segment_size=256, overlap=8, batch_size=1):
    """
    Melakukan pemisahan audio menggunakan model tertentu.
    """
    try:
        print(f"Loading model: {model_name}")
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=models_dir,
            output_dir=output_folder,
            output_format=output_format,
            use_autocast=use_autocast,
            output_single_stem=single_stem,
            mdxc_params={
                "segment_size": segment_size,
                "override_model_segment_size": True,
                "batch_size": batch_size,
                "overlap": overlap,
            }
        )
        model_path = roformer_models[model_name]
        separator.load_model(model_filename=model_path)

        print(f"\nSeparating audio: {input_file}\nUsing {model_name}")
        separator.separate(input_file)
        
        # Ambil nama file output dari direktori output
        output_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
        for file in output_files:
            if single_stem and single_stem.lower() in file.lower():
                return os.path.join(output_folder, file)

        raise RuntimeError(f"Expected output file with stem '{single_stem}' not found in {output_folder}")
    except Exception as e:
        raise RuntimeError(f"Error during separation with {model_name}: {e}")


def process_audio(args):
    """Pipeline lengkap untuk memproses audio."""
    # Temp folder untuk file sementara
    temp_dir = os.path.join(out_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Proses pertama: model pertama memisahkan vocal dan instrumental
    temp_vocal = perform_separation(
        input_file=args.input,
        model_name='MelBand Roformer Kim | FT by unwa',
        output_folder=temp_dir,
        output_format=args.output_format,
        single_stem='vocals',
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size
    )
    
    temp_instrument = [f for f in os.listdir(temp_dir) if 'instrumental' in f.lower()]
    temp_instrument = os.path.join(temp_dir, temp_instrument[0]) if temp_instrument else None

    # Proses kedua: model kedua memisahkan reverb dari vocal
    temp_noreverb = perform_separation(
        input_file=temp_vocal,
        model_name='MelBand Roformer | De-Reverb Less Aggressive by anvuew',
        output_folder=temp_dir,
        output_format=args.output_format,
        single_stem='noreverb',
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size
    )

    # Proses ketiga: model ketiga memisahkan noise dari vocal noreverb
    temp_clean = perform_separation(
        input_file=temp_noreverb,
        model_name='Mel-Roformer-Denoise-Aufr33-Aggr',
        output_folder=temp_dir,
        output_format=args.output_format,
        single_stem='dry',
        segment_size=args.segment_size,
        overlap=args.overlap,
        batch_size=args.batch_size
    )

    # Proses opsional: model keempat memisahkan crowd
    final_vocal = os.path.join(out_dir, "vocal_final.wav")
    if args.noCrowd:
        final_vocal = perform_separation(
            input_file=temp_clean,
            model_name='Mel-Roformer-Crowd-Aufr33-Viperx',
            output_folder=out_dir,
            output_format=args.output_format,
            single_stem='no_crowd',
            segment_size=args.segment_size,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
    else:
        os.rename(temp_clean, final_vocal)

    # Simpan hasil instrumental dari proses pertama
    final_instrumental = os.path.join(out_dir, "instrumental.wav")
    if temp_instrument:
        os.rename(temp_instrument, final_instrumental)

    # Bersihkan file sementara
    for temp_file in os.listdir(temp_dir):
        temp_file_path = os.path.join(temp_dir, temp_file)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    print(f"Processing complete! Files saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline untuk memisahkan audio menggunakan beberapa model Roformer.")
    parser.add_argument("--input", type=str, required=True, help="Path ke file audio input.")
    parser.add_argument("--output_format", type=str, default="wav", choices=["wav", "flac", "mp3", "ogg"], help="Format output audio.")
    parser.add_argument("--output_folder", type=str, default="./outputs", help="Direktori untuk menyimpan hasil output.")
    parser.add_argument("--segment_size", type=int, default=256, help="Ukuran segmen untuk pemrosesan.")
    parser.add_argument("--overlap", type=int, default=8, help="Jumlah overlap antar segmen.")
    parser.add_argument("--batch_size", type=int, default=1, help="Ukuran batch untuk pemrosesan.")
    parser.add_argument("--noCrowd", action="store_true", help="Aktifkan pemrosesan crowd/noCrowd.")

    args = parser.parse_args()

    # Update direktori output jika diperlukan
    out_dir = args.output_folder
    os.makedirs(out_dir, exist_ok=True)

    # Jalankan pipeline
    process_audio(args)
