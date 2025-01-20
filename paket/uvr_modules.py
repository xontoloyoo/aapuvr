import os
import traceback
import logging

logger = logging.getLogger(__name__)

import ffmpeg
import torch

from paket.config import Config
from paket.mdxnet import MDXNetDereverb
from paket.vr import AudioPre, AudioPreDeEcho

config = Config()

def uvr(model_name, dir_wav_input, save_root_vocal, wav_inputs, save_root_ins, agg, format0):
    infos = []
    try:
        pre_fun = None
        inp_root = dir_wav_input if dir_wav_input is not None else ""
        save_root_vocal = save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_ins = save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )

        is_hp3 = "HP3" in model_name

        if dir_wav_input:
            # Filter hanya file audio yang valid
            paths = [
                os.path.join(inp_root, name)
                for name in os.listdir(inp_root)
                if os.path.isfile(os.path.join(inp_root, name))
                and not name.startswith(".")  # Abaikan file/folder tersembunyi
                and name.lower().endswith((".wav", ".mp3", ".flac", ".m4a")) #Hanya memproses file berekstensi audio saja
            ]

        elif wav_inputs:
            paths = [wav_inputs]
        else:
            paths = []

        for path in paths:
            inp_path = path
            need_reformat = 0
            done = 0

            # Periksa apakah file audio valid menggunakan ffmpeg
            try:
                ffmpeg.probe(inp_path)
            except ffmpeg.Error as e:
                infos.append(f"Skipping invalid audio file: {inp_path} - {e.stderr.decode()}")
                yield "\n".join(infos)
                continue  # Lewati file ini dan lanjutkan ke file berikutnya

            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                    done = 1
                else:
                    need_reformat = 1
            except:
                need_reformat = 1
                traceback.print_exc()

            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path

            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except Exception as e:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)

    except Exception as e:
        infos.append(f"An error occurred: {e}")
        yield "\n".join(infos)
    finally:
        try:
            if pre_fun is not None:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
        except:
            traceback.print_exc()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")

        yield "\n".join(infos)
