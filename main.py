import os
import re
import json
import uuid
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import whisper
from yt_dlp import YoutubeDL

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "8"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
VOD_LANGUAGE = os.environ.get("VOD_LANGUAGE", "")  # e.g. "de"
MAX_SEGMENTS_KEEP = int(os.environ.get("MAX_SEGMENTS_KEEP", "30000"))

# Moment detection tuning
MOMENT_WINDOW_SEC = int(os.environ.get("MOMENT_WINDOW_SEC", "25"))
MOMENT_COOLDOWN_SEC = int(os.environ.get("MOMENT_COOLDOWN_SEC", "35"))

# Game detector keyword packs (editable)
GAME_KEYWORDS = {
    "valorant": ["ace","clutch","one tap","onetap","headshot","diff","ult","ultimate","defuse","spike","plant","retake"],
    "fortnite": ["one pump","boxed","box","cracked","shield","edit","piece","piece control","pump","zone","rotate"],
    "cs": ["one tap","headshot","ace","clutch","defuse","plant","smoke","flash","nade"],
}

HYPE_WORDS = ["no way","wtf","lets go","let's go","insane","crazy","holy","omg","gg","nice","sick","unreal","brooo","bruh"]

app = FastAPI(title="Twitch Transcriber Master+")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

jobs: Dict[str, dict] = {}

def seconds_to_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h{m:02d}m{s:02d}s"

def twitch_ts_link(url: str, seconds: float) -> str:
    ts = seconds_to_hms(seconds)
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={ts}"

def _safe_word_regex(q: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)

def _tokenize(text: str) -> List[str]:
    raw = re.findall(r"[\wäöüß']+", (text or "").lower())
    out = []
    for t in raw:
        t = re.sub(r"[^a-z0-9äöüß]+", "", t)
        if t and len(t) >= 2:
            out.append(t)
    return out

def job_emit(job_id: str, item: dict):
    jobs[job_id]["sse_q"].put(item)

def download_vod_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    tmpl = str(out_dir / f"{uid}_%(id)s.%(ext)s")
    ydl_opts = {"format":"bestaudio/best","outtmpl":tmpl,"quiet":True,"postprocessors":[]}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))

def transcribe_file(model, path: Path, offset_seconds: float = 0.0, language: str = "") -> List[dict]:
    kwargs = {"verbose": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(path), **kwargs)
    segs = []
    for s in result.get("segments", []):
        segs.append({
            "start": float(s["start"]) + offset_seconds,
            "end": float(s["end"]) + offset_seconds,
            "text": (s["text"] or "").strip(),
        })
    return segs

def _add_moment(job: dict, start_sec: float, label: str, reason: str):
    # cooldown to avoid spam
    now = start_sec
    if job["moments"]:
        if now - job["moments"][-1]["start"] < MOMENT_COOLDOWN_SEC:
            return
    m = {
        "start": start_sec,
        "time": seconds_to_hms(start_sec),
        "label": label,
        "reason": reason,
        "link": twitch_ts_link(job["url"], start_sec) if job["mode"] == "vod" else None,
    }
    job["moments"].append(m)
    job_emit(job["id"], {"type":"moment","moment":m})

def _detect_hype(text: str) -> int:
    t = (text or "").lower()
    score = 0
    for w in HYPE_WORDS:
        if w in t:
            score += 2
    score += t.count("!")  # punctuation excitement
    if "??" in t or "!!!" in t:
        score += 2
    return score

def _detect_game_hits(text: str) -> List[Tuple[str,str]]:
    t = (text or "").lower()
    hits = []
    for game, kws in GAME_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                hits.append((game, kw))
    return hits

def _apply_segment(job: dict, seg: dict):
    job["segments"].append(seg)
    if len(job["segments"]) > MAX_SEGMENTS_KEEP:
        job["segments"] = job["segments"][-MAX_SEGMENTS_KEEP:]

    # Frequency
    for tok in _tokenize(seg["text"]):
        job["freq"][tok] = job["freq"].get(tok, 0) + 1

    # Windowed activity for moment detection
    job["activity"].append((seg["start"], len(seg["text"]), _detect_hype(seg["text"])))
    # keep last N seconds
    cutoff = seg["start"] - MOMENT_WINDOW_SEC
    while job["activity"] and job["activity"][0][0] < cutoff:
        job["activity"].pop(0)

    # simple moment heuristic: lots of text or hype score in window
    total_chars = sum(x[1] for x in job["activity"])
    total_hype = sum(x[2] for x in job["activity"])
    if total_hype >= 6:
        _add_moment(job, max(0.0, seg["start"]-8), "HYPE Moment", f"hype-score={total_hype}")
    elif total_chars >= 650:
        _add_moment(job, max(0.0, seg["start"]-8), "Fast Talk", f"talk-burst chars={total_chars}")

    # game detector
    hits = _detect_game_hits(seg["text"])
    for game, kw in hits:
        job["game_hits"][game] = job["game_hits"].get(game, 0) + 1
        # auto-moment for big terms
        if kw in ("ace","clutch","one pump","boxed","headshot"):
            _add_moment(job, max(0.0, seg["start"]-6), f"{game.upper()} Moment", f"keyword='{kw}'")

    # Alerts (watchlist)
    for w in list(job["watch"]):
        pat = _safe_word_regex(w)
        if pat.search(seg["text"]):
            payload = {
                "type": "alert",
                "word": w,
                "time": seconds_to_hms(seg["start"]),
                "start": seg["start"],
                "text": seg["text"],
                "link": twitch_ts_link(job["url"], seg["start"]) if job["mode"]=="vod" else None,
            }
            job_emit(job["id"], payload)

def run_vod_job(job_id: str, url: str):
    job = jobs[job_id]
    job["status"] = "downloading"
    job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})

    try:
        audio_path = download_vod_audio(url, job["dir"] / "media")
    except Exception as e:
        job["status"] = "error"
        job_emit(job_id, {"type":"error","message":f"Download failed: {e}"})
        return

    job["status"] = "transcribing"
    job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})

    model = whisper.load_model(WHISPER_MODEL)
    try:
        segs = transcribe_file(model, audio_path, 0.0, language=VOD_LANGUAGE)
        for s in segs:
            _apply_segment(job, s)
            job_emit(job_id, {"type":"segment","segment":s})
        job["status"] = "done"
        job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})
    except Exception as e:
        job["status"] = "error"
        job_emit(job_id, {"type":"error","message":f"Transcription failed: {e}"})

def run_live_job(job_id: str, url: str):
    job = jobs[job_id]
    job["status"] = "capturing"
    job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})

    chunks_dir = job["dir"] / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(chunks_dir / "chunk_%08d.wav")

    streamlink_cmd = ["streamlink", url, "best", "-O"]
    ffmpeg_cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-i","pipe:0","-vn","-ac","1","-ar","16000",
        "-f","segment","-segment_time", str(SEGMENT_SECONDS),
        "-reset_timestamps","1", out_tmpl
    ]

    try:
        p1 = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(ffmpeg_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        job["procs"] = [p1, p2]
    except Exception as e:
        job["status"] = "error"
        job_emit(job_id, {"type":"error","message":f"Live capture failed: {e}"})
        return

    model = whisper.load_model(WHISPER_MODEL)
    last_idx = -1

    try:
        while not job.get("stop"):
            wavs = sorted(chunks_dir.glob("chunk_*.wav"))
            for w in wavs:
                m = re.search(r"chunk_(\d+)\.wav$", w.name)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx <= last_idx:
                    continue

                s1 = w.stat().st_size
                time.sleep(0.2)
                s2 = w.stat().st_size
                if s1 != s2:
                    continue

                offset = idx * SEGMENT_SECONDS
                segs = transcribe_file(model, w, offset, language="")
                for s in segs:
                    _apply_segment(job, s)
                    job_emit(job_id, {"type":"segment","segment":s})
                last_idx = idx

            p1, p2 = job.get("procs", [None, None])
            if p1 and p1.poll() is not None:
                job["status"] = "ended"
                job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})
                break
            if p2 and p2.poll() is not None:
                job["status"] = "ended"
                job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})
                break

            time.sleep(0.4)

        job["status"] = "stopped"
        job_emit(job_id, {"type":"status","status":job["status"],"mode":job["mode"]})
    except Exception as e:
        job["status"] = "error"
        job_emit(job_id, {"type":"error","message":f"Live transcription failed: {e}"})
    finally:
        for p in job.get("procs", []):
            try: p.terminate()
            except Exception: pass

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "env_whisper_model": WHISPER_MODEL,
        "env_segment_seconds": SEGMENT_SECONDS,
    })

@app.post("/api/start")
async def api_start(payload: dict):
    url = (payload.get("url") or "").strip()
    mode = (payload.get("mode") or "").strip().lower()
    if not url:
        raise HTTPException(400, "Missing url")
    if mode not in ("vod","live"):
        raise HTTPException(400, "mode must be 'vod' or 'live'")

    job_id = uuid.uuid4().hex[:10]
    job_dir = DATA / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    jobs[job_id] = {
        "id": job_id,
        "url": url,
        "mode": mode,
        "status": "queued",
        "dir": job_dir,
        "segments": [],
        "freq": {},
        "watch": set(),
        "moments": [],
        "activity": [],  # (t, chars, hype)
        "game_hits": {},
        "sse_q": queue.Queue(),
        "stop": False,
        "procs": [],
    }

    t = threading.Thread(target=run_live_job if mode=="live" else run_vod_job, args=(job_id, url), daemon=True)
    jobs[job_id]["thread"] = t
    t.start()

    return {"job_id": job_id}

@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    job["stop"] = True
    return {"ok": True}

@app.post("/api/watch/{job_id}")
async def api_watch(job_id: str, payload: dict):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    words = payload.get("words") or []
    if not isinstance(words, list):
        raise HTTPException(400, "words must be list")
    job["watch"] = set([w.strip() for w in words if isinstance(w, str) and w.strip()])
    return {"ok": True, "watch": sorted(list(job["watch"]))}

@app.get("/api/search/{job_id}")
async def api_search(job_id: str, q: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    q = (q or "").strip()
    if not q:
        return {"results": []}
    pat = _safe_word_regex(q)
    out = []
    for s in job["segments"]:
        if pat.search(s["text"]):
            out.append({
                "start": s["start"],
                "time": seconds_to_hms(s["start"]),
                "text": s["text"],
                "link": twitch_ts_link(job["url"], s["start"]) if job["mode"]=="vod" else None,
            })
    return {"results": out[:400]}

@app.get("/api/freq/{job_id}")
async def api_freq(job_id: str, top: int = 25):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    items = sorted(job["freq"].items(), key=lambda kv: kv[1], reverse=True)[:max(1, min(200, int(top)))]
    return {"items": [{"word": w, "count": c} for w, c in items]}

@app.get("/api/moments/{job_id}")
async def api_moments(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {"moments": job["moments"][-200:]}

@app.get("/api/analytics/{job_id}")
async def api_analytics(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    total_segments = len(job["segments"])
    total_words = sum(len(_tokenize(s["text"])) for s in job["segments"])
    top_games = sorted(job["game_hits"].items(), key=lambda kv: kv[1], reverse=True)
    return {
        "segments": total_segments,
        "words": total_words,
        "game_hits": [{"game": g, "hits": h} for g, h in top_games[:10]],
        "mode": job["mode"],
        "status": job["status"],
    }

@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def gen():
        q = job["sse_q"]
        yield f"data: {json.dumps({'type':'status','status':job['status'],'mode':job['mode']})}\n\n"
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"keepalive\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
