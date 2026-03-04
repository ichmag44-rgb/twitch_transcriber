import os
import re
import json
import uuid
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import whisper
from yt_dlp import YoutubeDL

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

SEGMENT_SECONDS = int(os.environ.get("SEGMENT_SECONDS", "10"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

app = FastAPI(title="Twitch Transcriber")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static")), name="static")

jobs: Dict[str, dict] = {}

def _safe_word_regex(q: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(q)}\b", re.IGNORECASE)

def seconds_to_hms(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}h{m:02d}m{s:02d}s"

def twitch_ts_link(original_url: str, seconds: float) -> str:
    ts = seconds_to_hms(seconds)
    joiner = "&" if "?" in original_url else "?"
    return f"{original_url}{joiner}t={ts}"

def download_vod_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    tmpl = str(out_dir / f"{uid}_%(id)s.%(ext)s")
    ydl_opts = {"format":"bestaudio/best","outtmpl":tmpl,"quiet":True,"postprocessors":[]}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))

def transcribe_file(model, path: Path, offset_seconds: float = 0.0):
    result = model.transcribe(str(path), verbose=False)
    segs = []
    for s in result.get("segments", []):
        segs.append({
            "start": float(s["start"]) + offset_seconds,
            "end": float(s["end"]) + offset_seconds,
            "text": s["text"].strip(),
        })
    return segs

def job_emit(job_id: str, item: dict):
    jobs[job_id]["sse_q"].put(item)

def run_vod_job(job_id: str, url: str):
    job = jobs[job_id]
    job["status"] = "downloading"
    job_emit(job_id, {"type":"status","status":job["status"]})
    try:
        audio_path = download_vod_audio(url, job["dir"] / "media")
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"Download failed: {e}"
        job_emit(job_id, {"type":"error","message":job["error"]})
        return

    job["status"] = "transcribing"
    job_emit(job_id, {"type":"status","status":job["status"]})

    model = whisper.load_model(WHISPER_MODEL)
    try:
        segs = transcribe_file(model, audio_path, 0.0)
        job["segments"].extend(segs)
        for s in segs:
            job_emit(job_id, {"type":"segment","segment":s})
        job["status"] = "done"
        job_emit(job_id, {"type":"status","status":job["status"]})
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"Transcription failed: {e}"
        job_emit(job_id, {"type":"error","message":job["error"]})

def run_live_job(job_id: str, url: str):
    job = jobs[job_id]
    job["status"] = "capturing"
    job_emit(job_id, {"type":"status","status":job["status"]})

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
        job["error"] = f"Live capture failed: {e}"
        job_emit(job_id, {"type":"error","message":job["error"]})
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

                # wait until stable
                s1 = w.stat().st_size
                time.sleep(0.2)
                s2 = w.stat().st_size
                if s1 != s2:
                    continue

                offset = idx * SEGMENT_SECONDS
                segs = transcribe_file(model, w, offset)
                job["segments"].extend(segs)
                for s in segs:
                    job_emit(job_id, {"type":"segment","segment":s})
                last_idx = idx

            p1, p2 = job.get("procs", [None, None])
            if p1 and p1.poll() is not None:
                job["status"] = "ended"
                job_emit(job_id, {"type":"status","status":job["status"]})
                break
            if p2 and p2.poll() is not None:
                job["status"] = "ended"
                job_emit(job_id, {"type":"status","status":job["status"]})
                break

            time.sleep(0.5)

        job["status"] = "stopped"
        job_emit(job_id, {"type":"status","status":job["status"]})
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"Live transcription failed: {e}"
        job_emit(job_id, {"type":"error","message":job["error"]})
    finally:
        for p in job.get("procs", []):
            try: p.terminate()
            except Exception: pass

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "env_whisper_model": WHISPER_MODEL})

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

    jobs[job_id] = {"id":job_id,"url":url,"mode":mode,"status":"queued","error":None,"dir":job_dir,"segments":[],
                    "sse_q": queue.Queue(),"stop":False,"procs":[]}

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
    return {"results": out[:200]}

@app.get("/api/events/{job_id}")
async def api_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    def gen():
        q = job["sse_q"]
        yield f"data: {json.dumps({'type':'status','status':job['status']})}\n\n"
        while True:
            try:
                item = q.get(timeout=15)
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"keepalive\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")