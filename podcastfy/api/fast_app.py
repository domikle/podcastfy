"""
FastAPI implementation for Podcastify podcast generation service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from ..client import generate_podcast
import uvicorn
import logging

logger = logging.getLogger("podcastfy")
logging.basicConfig(level=logging.INFO)


def load_base_config() -> Dict[Any, Any]:
    try:
        base_dir = Path(__file__).resolve().parents[1]  # /app/podcastfy
        config_path = base_dir / "conversation_config.yaml"
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        logger.warning("Could not load base config: %s", e)
        return {}


def merge_configs(base_config: Dict[Any, Any], user_config: Dict[Any, Any]) -> Dict[Any, Any]:
    merged = base_config.copy()

    if "text_to_speech" in merged and "text_to_speech" in user_config:
        merged["text_to_speech"].update(user_config.get("text_to_speech", {}))

    for key, value in user_config.items():
        if key != "text_to_speech":
            if value is not None:
                merged[key] = value

    return merged


app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)


def _set_env_if_present(env_name: str, value: Any) -> None:
    if value is not None and value != "":
        os.environ[env_name] = str(value)


def _set_env_alias_if_present(target: str, source: str) -> None:
    """
    If `target` is not set but `source` is set, copy it.
    Useful for users who already have ELEVENLABS_KEY etc.
    """
    if not os.getenv(target) and os.getenv(source):
        os.environ[target] = os.environ[source]


def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, list):
        out = []
        for x in v:
            cx = _clean_value(x)
            if cx is None:
                continue
            out.append(cx)
        return out if out else None
    if isinstance(v, dict):
        out = {k: _clean_value(val) for k, val in v.items()}
        out = {k: val for k, val in out.items() if val is not None}
        return out if out else None
    return v


def _parse_voice(value: Any) -> Optional[Union[str, Dict[str, str]]]:
    """
    Accept:
      - "Rachel" (name)
      - "21m00Tcm4TlvDq8ikWAM" (voice_id as string)
      - {"voice_id": "..."} or {"id": "..."} or {"name": "..."}
    Returns:
      - string (name or id) OR dict {"voice_id": "..."} when given as dict
    """
    if value is None:
        return None

    if isinstance(value, dict):
        vid = value.get("voice_id") or value.get("id")
        vname = value.get("name")
        if isinstance(vid, str) and vid.strip():
            return {"voice_id": vid.strip()}
        if isinstance(vname, str) and vname.strip():
            return vname.strip()
        return None

    if isinstance(value, str):
        s = value.strip()
        return s if s else None

    return None


@app.post("/generate")
def generate_podcast_endpoint(data: dict):
    try:
        # Keys nur setzen wenn vorhanden, sonst NoneType Crash
        _set_env_if_present("OPENAI_API_KEY", data.get("openai_key"))
        _set_env_if_present("GEMINI_API_KEY", data.get("google_key"))
        _set_env_if_present("GOOGLE_API_KEY", data.get("google_key"))  # manche Libs erwarten GOOGLE_API_KEY
        _set_env_if_present("ELEVENLABS_API_KEY", data.get("elevenlabs_key"))

        # ENV Alias: wenn User ELEVENLABS_KEY nutzt, mappen wir es automatisch
        _set_env_alias_if_present("ELEVENLABS_API_KEY", "ELEVENLABS_KEY")

        base_config = load_base_config()

        # output_language Alias: du sendest "language": "de"
        output_language = (
            data.get("output_language")
            or data.get("language")
            or base_config.get("output_language", "English")
        )

        tts_model = data.get(
            "tts_model",
            base_config.get("text_to_speech", {}).get("default_tts_model", "openai"),
        )
        tts_base_config = base_config.get("text_to_speech", {}).get(tts_model, {}) or {}

        voices_in = data.get("voices", {}) or {}
        default_voices = tts_base_config.get("default_voices", {}) or {}

        q_voice = _parse_voice(voices_in.get("question")) or _parse_voice(default_voices.get("question"))
        a_voice = _parse_voice(voices_in.get("answer")) or _parse_voice(default_voices.get("answer"))

        # Optional: wenn du ids als string schicken willst, kannst du das auch als dict erzwingen
        # Beispiel: {"voices": {"answer": {"voice_id": "..."}}}
        user_config = {
            "creativity": float(data.get("creativity", base_config.get("creativity", 0.7))),
            "conversation_style": data.get("conversation_style", base_config.get("conversation_style", [])),
            "roles_person1": data.get("roles_person1", base_config.get("roles_person1")),
            "roles_person2": data.get("roles_person2", base_config.get("roles_person2")),
            "dialogue_structure": data.get("dialogue_structure", base_config.get("dialogue_structure", [])),
            "podcast_name": data.get("name", base_config.get("podcast_name")),
            "podcast_tagline": data.get("tagline", base_config.get("podcast_tagline")),
            "output_language": output_language,
            "user_instructions": data.get("user_instructions", base_config.get("user_instructions", "")),
            "engagement_techniques": data.get("engagement_techniques", base_config.get("engagement_techniques", [])),
            "text_to_speech": {
                "default_tts_model": tts_model,
                "model": tts_base_config.get("model"),
                "default_voices": {
                    "question": q_voice,
                    "answer": a_voice,
                },
            },
        }

        conversation_config = merge_configs(base_config, user_config)

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

        # Inputs bereinigen: None, leere Strings, leere Listen/Dicts raus
        gp_kwargs = _clean_value(gp_kwargs) or {}
        if "conversation_config" not in gp_kwargs:
            gp_kwargs["conversation_config"] = conversation_config

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
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
