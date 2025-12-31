import os
import logging
from fastapi import FastAPI, HTTPException, Request

app = FastAPI(debug=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("podcastfy")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate")
async def generate_podcast_endpoint(request: Request):
    data = await request.json()
    logger.debug(f"Incoming request data: {data}")

    # ---------- INPUT ----------
    text = data.get("text")
    topic = data.get("topic")
    language = data.get("language", "en")

    if not text and not topic:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'text' or 'topic'."
        )

    prompt = text or f"Create a podcast about: {topic}"

    # ---------- API KEYS ----------
    openai_key = data.get("openai_key") or os.getenv("OPENAI_API_KEY")
    google_key = data.get("google_api_key") or os.getenv("GOOGLE_API_KEY")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        backend = "openai"
        logger.info("Using OpenAI backend")

    elif google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        backend = "gemini"
        logger.info("Using Gemini backend")

    else:
        raise HTTPException(
            status_code=400,
            detail="No AI API key provided (OpenAI or Gemini)."
        )

    # ---------- OUTPUT DIR ----------
    output_dir = os.getenv("OUTPUT_DIR", "/app/output")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- GENERATION ----------
    try:
        from podcastfy.core.generator import PodcastGenerator

        generator = PodcastGenerator(
            language=language,
            output_dir=output_dir
        )

        audio_path = generator.generate_from_text(prompt)

        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError("Audio generation failed or returned empty file")

        return {
            "status": "success",
            "backend": backend,
            "audio_file": audio_path
        }

    except Exception as e:
        logger.exception("Podcast generation failed")
        raise HTTPException(status_code=500, detail=str(e))