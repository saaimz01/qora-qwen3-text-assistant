"""FastAPI backend for the Local Text Assistant (Ollama + qwen3:4b).

Run:   uvicorn main:app --reload
Open:  http://127.0.0.1:8000
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, Field

MODEL = "qwen3:4b"
THINK = True            # reasoning is requested but never sent to the page; try False for faster replies
NUM_CTX = 8192          # context window; raise it if you work with very long texts
MAX_TEXT_CHARS = 20000

STATIC_DIR = Path(__file__).parent / "static"

client = AsyncClient()  # talks to Ollama at http://localhost:11434
app = FastAPI(title="Local Text Assistant")


# ---------------------------------------------------------------
# Hardcoded prompt templates
# ---------------------------------------------------------------
SUMMARIZE_TEMPLATE = """Summarize the text inside the <text> tags in at most {limit} words.
Focus only on the key points.

<text>{text}</text>"""

INFER_TEMPLATE = """Analyze the text inside the <text> tags.
Return ONLY valid JSON with these keys:
- "sentiment": "positive", "negative" or "neutral"
- "emotions": a list of up to 5 emotions expressed (empty list if none)
- "topics": a list of up to 5 short topics
- "is_angry": true or false
If a value cannot be determined, use "unknown". Output nothing except the JSON.

<text>{text}</text>"""

TRANSFORM_TEMPLATE = """Apply this instruction to the text inside the <text> tags: {instruction}.
Keep the original meaning. Output only the resulting text, with no explanations.

<text>{text}</text>"""

DEFAULT_WORD_LIMIT = 30
DEFAULT_TRANSFORM = "Proofread and correct the spelling and grammar"

# Low temperature = consistent output for the tools, a little more freedom for custom prompts
# Sent with every request, so the model has a language and a role even for short prompts
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running locally. "
    "Always reply in the same language as the user's message (English by default), "
    "unless the task asks for a different language, such as a translation. "
    "If you don't know the answer, say \"I don't know\"."
)

# Qwen3's authors recommend 0.6 in thinking mode and warn against 0 (greedy decoding)
TEMPERATURE = {"summarize": 0.6, "infer": 0.6, "transform": 0.6, "custom": 0.6}


# ---------------------------------------------------------------
# Request model and prompt building
# ---------------------------------------------------------------
class ChatRequest(BaseModel):
    mode: Literal["summarize", "infer", "transform", "custom"]
    text: str = Field(max_length=MAX_TEXT_CHARS)
    option: str = Field(default="", max_length=300)


def build_prompt(req: ChatRequest) -> str:
    text = req.text.strip()
    option = req.option.strip()

    if req.mode == "custom":
        return text  # the user's text is the whole prompt

    # Remove a closing tag so the user's text can't break out of the delimiters
    text = text.replace("</text>", "") # prompt injection defense

    if req.mode == "summarize":
        try:
            limit = min(max(int(option), 5), 500)
        except ValueError:
            limit = DEFAULT_WORD_LIMIT
        return SUMMARIZE_TEMPLATE.format(limit=limit, text=text)

    if req.mode == "infer":
        return INFER_TEMPLATE.format(text=text)

    return TRANSFORM_TEMPLATE.format(instruction=option or DEFAULT_TRANSFORM, text=text)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Please type something first.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(req)}]

    # Start the stream and wait for the first chunk here, so that "Ollama is down"
    # or "model missing" come back as a normal error the page can show.
    try:
        stream = await client.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            think=THINK,
            options={"temperature": TEMPERATURE[req.mode], "num_ctx": NUM_CTX},
        )
        chunks = stream.__aiter__()
        try:
            first = await chunks.__anext__()
        except StopAsyncIteration:
            first = None
    except ResponseError as e:
        if e.status_code == 404:
            raise HTTPException(status_code=503, detail=f"Model not found. Run: ollama pull {MODEL}")
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Could not reach Ollama. Start the Ollama app or run `ollama serve`.",
        )

    async def body():
        try:
            if first is not None and first.message.content:
                yield first.message.content
            async for chunk in chunks:
                if chunk.message.content:  # chunk.message.thinking (the reasoning) is ignored
                    yield chunk.message.content
        except Exception as e:
            yield f"\n\n[Stopped: {e}]"

    return StreamingResponse(
        body(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------
# Errors as plain text, which is what the page displays
# ---------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_error(request, exc: HTTPException):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc: RequestValidationError):
    return PlainTextResponse(
        f"Invalid request. Keep the text under {MAX_TEXT_CHARS} characters.", status_code=422
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)