# Qora - A Local Text Assistant

A local text assistant powered by **Qwen3 4B** through the **Ollama Python API**, served by a **FastAPI** backend that streams replies to a plain HTML page. It runs entirely on your own machine: no cloud API, no API key, no data leaving your computer.

It offers four modes: **Summarize**, **Infer** (sentiment, emotions and topics as JSON), **Transform** (translate, proofread, change tone) and **Custom prompt**.

## Project focus

This project was built to practice the techniques from DeepLearning.AI's [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) course, using a local model instead of OpenAI's API.

The effort went into two things:

1. **Implementing the Qwen API.** Calling `qwen3:4b` through Ollama's async Python client: streaming, reasoning (thinking) mode, model options, and error handling.
2. **Building the FastAPI backend.** Where the course ideas live: hardcoded prompt templates, delimiters, structured JSON output, input validation and a streaming endpoint.

**The frontend is deliberately plain HTML**, with no CSS and no framework. It is only there to give the backend something to talk to. All of the interesting logic is in `main.py`.

## Modes

| Mode | What it does | Option field |
|---|---|---|
| Summarize | Shortens the text to a word limit | Max words (default 30, clamped to 5-500) |
| Infer | Returns sentiment, emotions, topics and an anger flag as JSON | None |
| Transform | Translates, changes tone, or proofreads | What to do (blank = proofread) |
| Custom prompt | Sends whatever you type as the prompt | None |

Each request is independent: the assistant does not remember earlier messages.

## Course concepts and where they appear

| Course idea | Where it appears |
|---|---|
| Use delimiters to separate instructions from text | Templates wrap the user's text in `<text>...</text>` tags, and a typed `</text>` is removed so it can't break out |
| Ask for structured output | Infer mode demands JSON with fixed keys, and the page pretty-prints it |
| Check conditions and handle missing information | Infer says to use `"unknown"` when a value can't be determined; Transform defaults to proofreading when no instruction is given |
| Summarizing | Summarize template with a configurable word limit |
| Inferring | Infer template (sentiment, emotions, topics) |
| Transforming | Transform template (translation, tone, grammar) |
| Iterating on prompts | Templates are constants at the top of `main.py` and are easy to edit and re-test |
| Roles and system messages | A system prompt is sent with every request, followed by the user message |
| Temperature | Set per mode in the `TEMPERATURE` dictionary and passed to Ollama |

The course's *Expanding* lesson (drafting replies) is not included in this version.

## How it works

```
Browser (static/index.html)
   |  POST /api/chat   {"mode": "...", "text": "...", "option": "..."}
   v
FastAPI (main.py)
   |  validate -> build prompt from template -> add system prompt
   v
Ollama (AsyncClient, qwen3:4b, stream=True, think=True)
   |  chunks
   v
FastAPI streams only the answer text back; the model's reasoning is dropped
   v
Browser shows the reply as it arrives
```

### API

`POST /api/chat`

Request body (JSON):

| Field | Type | Notes |
|---|---|---|
| `mode` | `"summarize"` \| `"infer"` \| `"transform"` \| `"custom"` | Anything else is rejected |
| `text` | string | Required, up to 20,000 characters. In `custom` mode this is the whole prompt |
| `option` | string | Optional, up to 300 characters. Word limit for summarize, instruction for transform, ignored otherwise |

The response is streamed `text/plain` (UTF-8). Errors are plain text too:

| Status | Meaning |
|---|---|
| 400 | Empty text |
| 422 | Invalid request (unknown mode, text too long) |
| 502 | Ollama returned an error |
| 503 | Ollama is not running, or `qwen3:4b` is not downloaded |

`GET /` serves the page. FastAPI also generates an interactive API page at `/docs`, handy for testing the endpoint without the HTML frontend.

### Calling Qwen through Ollama

```python
stream = await client.chat(
    model="qwen3:4b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    stream=True,
    think=True,
    options={"temperature": 0.6, "num_ctx": 8192},
)
```

- **Async client:** `AsyncClient` keeps the server responsive while a reply is generating.
- **Streaming:** the reply is sent chunk by chunk, so text appears as it is produced.
- **Thinking mode:** Qwen3 is a reasoning model. With `think=True`, its reasoning arrives in a separate field (`message.thinking`) and only `message.content` is forwarded to the page.
- **Options:** temperature per mode, and a larger context window (`num_ctx`) so long texts are not silently truncated.

### Backend design decisions

- **First chunk is read before the response starts.** Once a streaming response begins, the HTTP status is already sent. Reading the first chunk inside a `try` block means "Ollama is down" or "model missing" can be returned as a proper error instead of a broken stream.
- **Prompt templates live on the server.** The page only picks a mode. Users can't change the instructions, and all prompts are in one place.
- **Input is validated** with a Pydantic model. The word limit is clamped, and braces in user text are safe with `str.format`.
- **Errors are returned as readable plain text** that the page can show directly.
- **Same-origin serving.** FastAPI serves the page itself, so no CORS configuration is needed.

## Project structure

```
text-assistant/
├── main.py            # FastAPI backend: Ollama calls, templates, validation, streaming
├── requirements.txt
├── README.md
└── static/
    └── index.html     # plain HTML frontend (no CSS)
```

## Setup

**Requirements:** Python 3.10 or newer, and a recent [Ollama](https://ollama.com/download) (the `think` option needs a recent version).

```bash
# 1. Download the model (about 2.5 GB)
ollama pull qwen3:4b

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Start the server (run this from the folder that contains main.py)
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. Make sure Ollama is running first.

## Configuration

Settings are constants at the top of `main.py`:

| Constant | Default | Purpose |
|---|---|---|
| `MODEL` | `qwen3:4b` | Ollama model name |
| `THINK` | `True` | Ask the model to reason first; set `False` for faster replies |
| `NUM_CTX` | `8192` | Context window in tokens |
| `MAX_TEXT_CHARS` | `20000` | Longest accepted text |
| `SYSTEM_PROMPT` | see file | Sent with every request |
| `TEMPERATURE` | `0.6` for all modes | Randomness per mode |

## Lessons learned

- **Replies in Chinese.** A Custom prompt like "who are you" with no system message got a Chinese reply. The other modes were fine because their English templates set the language. Fix: send a system prompt with every request that tells the model to answer in the user's language.
- **Temperature 0 is a bad fit for Qwen3's thinking mode.** The [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B/blob/main/README.md) recommends 0.6 in thinking mode and warns that greedy decoding can cause endless repetitions, so all modes use 0.6.
- **Reasoning text leaking into replies.** On the author's setup, `think=False` still returned the reasoning (including a stray `</think>`) inside `content`. Using `think=True` and forwarding only `content` keeps it out of the page.
- **Slow replies.** Thinking mode generates a long hidden chain of reasoning before the answer. Setting `THINK = False` is the first thing to try if responses feel slow.
- **Silent truncation.** Ollama's default context window is small, so long inputs can lose their beginning without any error. `NUM_CTX` and `MAX_TEXT_CHARS` are set to avoid this.

## Troubleshooting

- **`Could not import module "main"`**: run `uvicorn main:app --reload` from the folder that contains `main.py`, and check the file isn't named `main.py.txt`.
- **"Could not reach Ollama"**: start the Ollama app, or run `ollama serve`.
- **"Model not found"**: run `ollama pull qwen3:4b`.
- **Error about the `think` argument**: update the library with `pip install -U ollama`, and update Ollama itself.

## Limitations and ideas

- No conversation memory: each request is independent. Adding it would mean the page sending the message history and the backend adding it to `messages`.
- No authentication or rate limiting, since it is meant for local use.
- The *Expanding* lesson (reply drafting) could be added as a fifth mode.
- Automated tests for the prompt-building and error paths.
- A styled frontend, if wanted. The backend API would not need to change.

## Acknowledgements

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) by DeepLearning.AI, taught by Isa Fulford and Andrew Ng
- [Ollama](https://ollama.com) and the [Qwen3](https://huggingface.co/Qwen/Qwen3-4B) model family
- [FastAPI](https://fastapi.tiangolo.com)
