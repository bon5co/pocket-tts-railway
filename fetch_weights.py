"""Download the Pocket TTS English weights, tokenizer and licensed voices into the Hugging Face cache at image build.

A download at boot would sit inside Railway's healthcheck window, so everything is part of the image and the server
runs with HF_HUB_OFFLINE=1.

The package's English config names two weight files: `kyutai/pocket-tts` (gated, includes the voice-cloning encoder)
and `kyutai/pocket-tts-without-voice-cloning` (ungated, `gated: false`, cc-by-4.0). It tries the first and falls back
to the second when the download fails. An anonymous build gets 401 on the first, so it lands on the second, and this
script asserts that, so the image can never silently contain the gated weights.

Only voices whose source recordings are CC0 or CC-BY are fetched (kyutai/tts-voices README, read 2026-11-25):
voice-donations (CC0), voice-zero (CC0), vctk (CC-BY 4.0), alba-mackenna (CC-BY 4.0). `cosette` (Expresso) and
`jean` (EARS) are CC-BY-NC, non-commercial, and are left out.
"""
import os
import sys

from huggingface_hub import hf_hub_download

from pocket_tts import TTSModel
from pocket_tts.utils.utils import get_predefined_voice

VOICES = [
    # CC0: Unmute voice donations, Voice-Zero (LibriVox)
    "marius", "javert", "bill_boerst", "peter_yearsley", "stuart_bell", "caro_davy",
    # CC-BY 4.0: Alba MacKenna, CSTR VCTK
    "alba", "anna", "vera", "fantine", "charles", "paul", "eponine", "azelma", "george", "mary", "jane",
    "michael", "eve",
]

model = TTSModel.load_model("english")
if model.has_voice_cloning:
    sys.exit("FATAL: the gated voice-cloning weights were loaded; this image must carry the ungated weights only")
print(f"model loaded, has_voice_cloning={model.has_voice_cloning}, sample_rate={model.sample_rate}", flush=True)

for name in VOICES:
    ref = get_predefined_voice(language="english", name=name).removeprefix("hf://")
    repo = "/".join(ref.split("/")[:2])
    rest = "/".join(ref.split("/")[2:])
    filename, revision = rest.split("@")
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    print(f"{os.path.getsize(path):>10} {name} {path}", flush=True)
