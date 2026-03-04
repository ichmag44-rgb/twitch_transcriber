# Twitch Transcriber Master++ (innovative pack)

## New features added
- AI Moment Detection (heuristics: hype words, talk bursts, game keywords)
- Game Detector (Valorant/Fortnite/CS keyword signals)
- Streamer Analytics (segments, words, game signal counters)
- Alerts + Instant Search + Live transcription

## Deploy (Render Docker)
- Push repo to GitHub
- Render: New Web Service -> Docker -> Clear build cache & deploy

## Speed (≈10x feel)
Set env vars in Render:
- WHISPER_MODEL=tiny
- VOD_LANGUAGE=de (if German)
- SEGMENT_SECONDS=6..10

## Notes
- "Auto Clip Generator": true Twitch clip creation needs Twitch OAuth + API. This version focuses on auto-moments + timestamp links.
