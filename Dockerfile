# Kyutai Pocket TTS API (100M parameters, CPU), Bearer API_KEY, OpenAI-compatible /v1/audio/speech.
# Source: https://github.com/bon5co/pocket-tts-railway
# CPU only (torch CPU wheel, no CUDA libraries), weights baked in (about 330 MB), stateless (no volume needed).
# upstream: kyutai-labs/pocket-tts (PyPI pocket-tts==4.0.1, MIT, the newest release as of 2026-11-25, published
# 2026-10-21). Weights: kyutai/pocket-tts-without-voice-cloning at commit a7056f505a4a8ca72d8b5fb1440071d6f1a35739
# (CC-BY-4.0, ungated), tokenizer at 00eac05ed3d16bdc3f6b5d598874019c34a89214; see fetch_weights.py.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip list 2>/dev/null | grep -iE "^(torch|pocket-tts|numpy|soundfile|fastapi|uvicorn) "

# Weights, tokenizer and licensed voices into a Hugging Face cache inside the image (see fetch_weights.py).
ENV HF_HOME=/models/hf HF_HUB_DISABLE_TELEMETRY=1
COPY fetch_weights.py /tmp/fetch_weights.py
RUN python -I /tmp/fetch_weights.py && rm -rf /models/hf/xet /models/hf/token /root/.cache && du -sh /models

# MIT licence of the upstream code, the CC-BY-4.0 legal text of the weights and voices, and this repository's notice.
COPY LICENSE-pocket-tts /licenses/pocket-tts-LICENSE
COPY LICENSE-CC-BY-4.0 /licenses/CC-BY-4.0-LICENSE
COPY NOTICE /licenses/NOTICE

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin tts \
    && chmod -R a+rX /models /licenses
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh

ENV HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
