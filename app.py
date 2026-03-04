"""
This module defines a simple Flask application that can download a Twitch video,
transcribe the audio and then search for occurrences of a given word in the
transcription.  The code relies on `yt_dlp` to fetch the media and
`openai_whisper` to perform the speech‑to‑text conversion.

The typical usage flow is:

1.  A visitor lands on the home page and enters a video URL.  The server
    downloads the audio track of that video and begins transcription.
2.  Once the transcription has finished the visitor is presented with a
    search form where they can enter any keyword.  The application will then
    locate every segment of the transcript that contains the keyword and
    return a list of timestamps along with direct links to the original
    video starting at those positions.

The project is intentionally lightweight and avoids external state.  Each
transcript is stored as a JSON file under a temporary directory and is
referenced via a unique identifier stored in the session.  You are free to
extend this example to include user accounts, a database or asynchronous
processing via a task queue such as Celery.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from flask import Flask, render_template, request, redirect, url_for, session, flash

try:
    from yt_dlp import YoutubeDL
except ImportError:
    # yt_dlp is not installed; leave a placeholder import so that the code
    # remains syntactically valid.  The functions relying on yt_dlp will
    # raise a RuntimeError if invoked without the dependency.
    YoutubeDL = None

try:
    import whisper
except ImportError:
    whisper = None  # type: ignore


app = Flask(__name__)
# A secret key is required for session management.  In a production
# deployment you should set this via an environment variable instead of
# hard‑coding it.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))

# Directory where downloaded media and transcripts will be stored.  This
# location is relative to the application root so that it stays within the
# project tree.  The directory will be created on demand.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class TranscriptSegment:
    """Represents a single segment of the transcript."""

    start: float
    end: float
    text: str


def ensure_data_dir() -> None:
    """Create the data directory if it does not already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def download_audio(url: str) -> str:
    """Download the best available audio stream from a Twitch video.

    Parameters
    ----------
    url:
        The full link to the video on Twitch.  This may be a VOD or a clip.

    Returns
    -------
    str
        The file system path to the downloaded audio file.

    Raises
    ------
    RuntimeError
        If yt_dlp is not available or downloading fails.
    """
    if YoutubeDL is None:
        raise RuntimeError(
            "yt_dlp must be installed to download audio.  Install it with 'pip install yt‑dlp'."
        )

    ensure_data_dir()
    # Use a unique identifier for the output file to avoid clashes.  yt_dlp
    # replaces %(id)s with the internal ID of the video, but we still add
    # a random prefix to guarantee uniqueness across multiple downloads.
    uid = uuid.uuid4().hex
    output_template = os.path.join(DATA_DIR, f"{uid}_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        # Leave the file as is; we will let whisper convert it on the fly
        "postprocessors": [],
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)
        return file_path


def transcribe_audio(path: str) -> List[TranscriptSegment]:
    """Transcribe the given audio file using OpenAI's Whisper model.

    Parameters
    ----------
    path:
        Absolute path to the audio or video file on the file system.

    Returns
    -------
    list of TranscriptSegment
        All segments produced by the model containing start time, end time and
        transcript text.

    Raises
    ------
    RuntimeError
        If the whisper module is not available.
    """
    if whisper is None:
        raise RuntimeError(
            "openai‑whisper must be installed to perform transcription.  Install it with 'pip install openai‑whisper'."
        )
    # Load a small model to balance speed and accuracy.  Adjust the size
    # according to your available compute resources.
    model = whisper.load_model("base")
    result = model.transcribe(path, verbose=False)
    segments = []
    for seg in result.get("segments", []):
        segments.append(TranscriptSegment(start=float(seg["start"]), end=float(seg["end"]), text=seg["text"]))
    return segments


def save_transcript(segments: List[TranscriptSegment]) -> str:
    """Write the transcript segments to a JSON file and return its identifier.

    The JSON file will contain a list of dictionaries with `start`, `end` and
    `text` fields.  The file is stored in the data directory using a UUID
    prefix.  The returned value is the UUID which can later be used to
    retrieve the file.
    """
    ensure_data_dir()
    uid = uuid.uuid4().hex
    path = os.path.join(DATA_DIR, f"{uid}.json")
    with open(path, "w", encoding="utf‑8") as fh:
        json.dump([seg.__dict__ for seg in segments], fh)
    return uid


def load_transcript(uid: str) -> List[TranscriptSegment]:
    """Load a transcript from disk using its UUID.

    Parameters
    ----------
    uid:
        The unique identifier returned by `save_transcript`.

    Returns
    -------
    list of TranscriptSegment
        The transcript segments read from the JSON file.
    """
    path = os.path.join(DATA_DIR, f"{uid}.json")
    with open(path, "r", encoding="utf‑8") as fh:
        data = json.load(fh)
    return [TranscriptSegment(**d) for d in data]


def find_word_occurrences(segments: List[TranscriptSegment], word: str) -> List[Tuple[float, str]]:
    """Locate all occurrences of a word within the transcript segments.

    Parameters
    ----------
    segments:
        A list of transcript segments as returned by `transcribe_audio` or
        `load_transcript`.
    word:
        The keyword to search for.  The search is case‑insensitive and will
        match whole words only.  For example, searching for "art" will not
        match "party".

    Returns
    -------
    list of (float, str)
        A list of tuples containing the start time in seconds and the
        corresponding line of text where the word was found.
    """
    occurrences = []
    # Build a regular expression that matches the word as a standalone token
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for seg in segments:
        if pattern.search(seg.text):
            occurrences.append((seg.start, seg.text))
    return occurrences


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds into a Twitch time string (e.g., 1h23m45s)."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes:02d}m")
    parts.append(f"{secs:02d}s")
    return "".join(parts)


@app.route("/")
def index():
    """Render the home page with the URL submission form."""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Handle submission of a video URL and start transcription."""
    url = request.form.get("video_url")
    if not url:
        flash("Bitte gib die Adresse des Videos an.")
        return redirect(url_for("index"))
    try:
        audio_path = download_audio(url)
        segments = transcribe_audio(audio_path)
        uid = save_transcript(segments)
    except Exception as exc:
        flash(f"Beim Verarbeiten ist ein Fehler aufgetreten: {exc}")
        return redirect(url_for("index"))
    # Store both the UUID and the original video link so that we can build
    # timestamped links later.
    session["transcript_id"] = uid
    session["video_link"] = url
    return render_template("search.html")


@app.route("/search", methods=["POST"])
def search():
    """Perform a keyword search in a previously generated transcript."""
    uid = session.get("transcript_id")
    video_link = session.get("video_link")
    if not uid or not video_link:
        flash("Es ist kein Transkript vorhanden. Bitte zunächst ein Video verarbeiten.")
        return redirect(url_for("index"))
    word = request.form.get("keyword")
    if not word:
        flash("Bitte gib ein Suchwort ein.")
        return redirect(url_for("search"))
    segments = load_transcript(uid)
    matches = find_word_occurrences(segments, word)
    # Construct a list of result objects containing time strings and links
    results = []
    for start_time, text in matches:
        hms = seconds_to_hms(start_time)
        # Append the t= query parameter to the original link to jump to the time
        if "?" in video_link:
            link = f"{video_link}&t={hms}"
        else:
            link = f"{video_link}?t={hms}"
        results.append({"time": hms, "text": text, "link": link})
    return render_template("results.html", results=results, word=word)


if __name__ == "__main__":
    # When running directly via `python app.py` this will start the
    # development server.  In production you should use a WSGI server
    # such as gunicorn or uWSGI and configure environment variables for
    # better performance and security.
    app.run(debug=True)