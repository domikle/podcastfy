from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

from ..client import generate_podcast

logger = logging.getLogger("podcastfy")
logging.basicConfig(level=logging.INFO)

app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)


def _set_env_if_present(env_name: str, value: Any) -> None:
    if value is not None and str(value).strip() != "":
        os.environ[env_name] = str(value).strip()


def _set_env_alias_if_present(target: str, source: str) -> None:
    if not os.getenv(target) and os.getenv(source):
        os.environ[target] = os.environ[source]


def load_base_config() -> Dict[Any, Any]:
    """
    Load /app/podcastfy/conversation_config.yaml
    """
    try:
        base_dir = Path(__file__).resolve().parents[1]  # /app/podcastfy
        config_path = base_dir / "conversation_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load base config: %s", e)
        return {}


def _parse_voice(value: Any) -> Optional[Union[str, Dict[str, str]]]:
    """
    Accept:
      - "Rachel" (voice name)
      - "uUnmYv9aJqaqzs1wcFRH" (voice_id as plain string)
      - {"voice_id": "..."} / {"id": "..."} / {"name": "..."}
    Return:
      - {"voice_id": "..."} for ids
      - "Name" for names
    """
    if value is None:
        return None

    if isinstance(value, dict):
        vid = value.get("voice_id") or value.get("id")
        if isinstance(vid, str) and vid.strip():
            return {"voice_id": vid.strip()}
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # treat id-like strings as voice_id
        if " " not in s and len(s) >= 10:
            return {"voice_id": s}
        return s

    return None


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge b into a (recursive dict merge)
    """
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _strip_empty(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    if isinstance(x, list):
        cleaned = []
        for item in x:
            ci = _strip_empty(item)
            if ci is not None:
                cleaned.append(ci)
        return cleaned if cleaned else None
    if isinstance(x, dict):
        cleaned = {k: _strip_empty(v) for k, v in x.items()}
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned if cleaned else None
    return x


@app.post("/generate")
def generate_podcast_endpoint(data: dict):
    try:
        # --- API keys ---
        _set_env_if_present("OPENAI_API_KEY", data.get("openai_key"))
        _set_env_if_present("GEMINI_API_KEY", data.get("google_key"))
        _set_env_if_present("GOOGLE_API_KEY", data.get("google_key"))
        _set_env_if_present("ELEVENLABS_API_KEY", data.get("elevenlabs_key"))

        # allow ELEVENLABS_KEY
        _set_env_alias_if_present("ELEVENLABS_API_KEY", "ELEVENLABS_KEY")

        base_config = load_base_config()

        output_language = (
            data.get("output_language")
            or data.get("language")
            or base_config.get("output_language", "English")
        )

        # --- tts model ---
        tts_model = data.get(
            "tts_model",
            (base_config.get("text_to_speech", {}) or {}).get("default_tts_model", "openai"),
        )

        # --- voices: model specific override ---
        tts_root = base_config.get("text_to_speech", {}) or {}
        tts_model_cfg = (tts_root.get(tts_model, {}) or {})

        voices_in = data.get("voices", {}) or {}
        default_voices = (tts_model_cfg.get("default_voices", {}) or {})

        q_voice = _parse_voice(voices_in.get("question")) or _parse_voice(default_voices.get("question"))
        a_voice = _parse_voice(voices_in.get("answer")) or _parse_voice(default_voices.get("answer"))

        # Build conversation_config override
        convo_override = {
            "podcast_name": data.get("name") or data.get("podcast_name") or base_config.get("podcast_name"),
            "podcast_tagline": data.get("tagline") or base_config.get("podcast_tagline"),
            "output_language": output_language,
            "user_instructions": data.get("user_instructions") or base_config.get("user_instructions", ""),
            "creativity": float(data.get("creativity", base_config.get("creativity", 0.7))),
            "conversation_style": data.get("conversation_style", base_config.get("conversation_style", [])),
            "roles_person1": data.get("roles_person1", base_config.get("roles_person1")),
            "roles_person2": data.get("roles_person2", base_config.get("roles_person2")),
            "dialogue_structure": data.get("dialogue_structure", base_config.get("dialogue_structure", [])),
            "engagement_techniques": data.get("engagement_techniques", base_config.get("engagement_techniques", [])),
            "text_to_speech": {
                "default_tts_model": tts_model,
                tts_model: {
                    **tts_model_cfg,
                    "default_voices": {
                        "question": q_voice,
                        "answer": a_voice,
                    },
                },
            },
        }

        conversation_config = _deep_merge(base_config, convo_override)
        conversation_config = _strip_empty(conversation_config) or conversation_config

        # --- inputs to generate_podcast ---
        gp_kwargs = {
            "urls": data.get("urls"),
            "url_file": data.get("url_file"),
            "transcript_file": data.get("transcript_file"),
            "image_paths": data.get("image_paths"),
            "text": data.get("text"),
            "topic": data.get("topic"),
            "conversation_config": conversation_config,
            "tts_model": tts_model,
            "longform": bool(data.get("is_long_form", False)),
        }
        gp_kwargs = _strip_empty(gp_kwargs) or {}

        logger.info(
            "generate_podcast inputs: %s",
            {k: ("<set>" if v else v) for k, v in gp_kwargs.items() if k != "conversation_config"},
        )

        result = generate_podcast(**gp_kwargs)

        if isinstance(result, str) and os.path.isfile(result):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result, output_path)
            return {"audioUrl": f"/audio/{filename}"}

        if hasattr(result, "audio_path") and isinstance(result.audio_path, str) and os.path.isfile(result.audio_path):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result.audio_path, output_path)
            return {"audioUrl": f"/audio/{filename}"}

        raise HTTPException(status_code=500, detail="Invalid result format")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /generate")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/health")
def healthcheck():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
