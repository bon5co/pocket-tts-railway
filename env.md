# Pocket TTS API - environment

Paste into Railway's variables panel. Everything else lives in the image, because a template variable holding a literal is dropped at generate time and republishes as a blank required field.

```
API_KEY="${{secret(32)}}"
```

## Optional

| Variable | Default | Values |
|---|---|---|
| `DEFAULT_VOICE` | `alba` | Voice used when a request names none: any id from `GET /v1/audio/voices` |
| `MAX_INPUT_CHARS` | `5000` | Longest `input` accepted (413 above it) |
| `MAX_SECONDS` | `240` | Wall-clock budget from request arrival; past it the request answers 504 (Railway's edge cuts a connection at about 5 minutes) |
| `MAX_PENDING` | `3` | Requests running or waiting; another answers 429 with `Retry-After: 30` |

## Set by the entrypoint at boot

`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are pinned to 1: the model is fastest on one thread and a host-sized pool oversubscribes the container.

`PORT` is left to Railway: the healthcheck dials the injected port, so the app honours it.
