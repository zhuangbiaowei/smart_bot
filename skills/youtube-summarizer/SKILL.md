---
name: youtube-summarizer
description: Automatically fetch YouTube video transcripts, generate structured summaries, and send full transcripts to messaging platforms. Detects YouTube URLs and provides metadata, key insights, and downloadable transcripts. This skill is for SUMMARIZING and TRANSCRIBING videos, NOT for downloading them.
version: 1.0.0
author: abe238
homepage: https://github.com/yt-dlp/yt-dlp
metadata:
  openclaw:
    emoji: "📺"
    skillKey: "youtube-summarizer"
    requires:
      bins: ["yt-dlp"]
      env: ["PATH"]
    install:
      - kind: python
        package: yt-dlp
        label: "Install yt-dlp (pip install yt-dlp)"
---

# YouTube Summarizer Skill

Automatically fetch transcripts from YouTube videos, generate structured summaries, and deliver full transcripts to messaging platforms.

## When to Use

Activate this skill when:
- User shares a YouTube URL (youtube.com/watch, youtu.be, youtube.com/shorts)
- User asks to summarize or transcribe a YouTube video
- User requests information about a YouTube video's content

## Dependencies

**Required:** yt-dlp must be installed and available in PATH.

Check if yt-dlp is available:
```bash
which yt-dlp || echo "yt-dlp not found"
```

If not installed, you can install it:
```bash
pip3 install yt-dlp
```

## Workflow

### 1. Detect YouTube URL
Extract video ID from these patterns:
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`
- Direct video ID: `VIDEO_ID` (11 characters)

### 2. Fetch Video Metadata
Get video information using yt-dlp:
```bash
yt-dlp --dump-json --skip-download "VIDEO_URL" 2>/dev/null | head -c 10000
```

Extract from the JSON:
- `title` - Video title
- `uploader` - Channel name
- `view_count` - View count
- `upload_date` - Publication date (YYYYMMDD format)
- `duration` - Video duration in seconds

### 3. Fetch Transcript

**Option A: Use yt-dlp to get auto-generated subtitles (if available)**
```bash
yt-dlp --list-subs --skip-download "VIDEO_URL" 2>/dev/null
```

Download subtitles if available:
```bash
yt-dlp --write-auto-sub --sub-lang en --skip-download --sub-format txt -o "/tmp/yt_%(id)s.%(ext)s" "VIDEO_URL" 2>/dev/null
```

Read the subtitle file:
```bash
cat /tmp/yt_VIDEO_ID.en.txt 2>/dev/null || cat /tmp/yt_VIDEO_ID.txt 2>/dev/null
```

**Option B: If yt-dlp subtitles fail, inform user:**
> "I couldn't retrieve the transcript for this video. This may be because:
> - The video doesn't have captions/subtitles enabled
> - The video owner has disabled auto-generated captions
> - The video is restricted or private
>
> You can try:
> - Using YouTube's built-in transcript feature (click '...' below the video → 'Show transcript')
> - Trying a different video that has captions enabled"

### 4. Process the Data

Clean up the transcript text:
- Remove timestamps if present
- Remove duplicate lines
- Join lines into paragraphs

Extract full text for summarization.

### 5. Generate Summary

Create a structured summary using this template:

```markdown
📹 **Video:** [title]
👤 **Channel:** [author] | 👁️ **Views:** [views] | 📅 **Published:** [date]

**🎯 Main Thesis:**
[1-2 sentence core argument/message]

**💡 Key Insights:**
- [insight 1]
- [insight 2]
- [insight 3]
- [insight 4]
- [insight 5]

**📝 Notable Points:**
- [additional point 1]
- [additional point 2]

**🔑 Takeaway:**
[Practical application or conclusion]
```

Aim for:
- Main thesis: 1-2 sentences maximum
- Key insights: 3-5 bullets, each 1-2 sentences
- Notable points: 2-4 supporting details
- Takeaway: Actionable conclusion

### 6. Save Full Transcript (Optional)

If the user wants the full transcript saved:

Create the transcripts directory:
```bash
mkdir -p ~/.smart_bot/transcripts
```

Save the transcript:
```bash
cat > ~/.smart_bot/transcripts/YYYY-MM-DD_VIDEO_ID.txt << 'EOF'
Video: [title]
Channel: [author]
URL: [video_url]
Date: [date]

---

[Full transcript text]
EOF
```

### 7. Reply with Summary

Send the structured summary as your response to the user.

If you saved the transcript file, mention its location.

## Error Handling

**If yt-dlp is not installed:**
```bash
# Try to install it
pip3 install yt-dlp
```

**If transcript fetch fails:**
- Check if video has captions enabled
- Try with `--sub-lang en` fallback if requested language unavailable
- Inform user that transcript is not available and suggest alternatives:
  - Manual YouTube transcript feature
  - Video may not have captions
  - Try a different video

**If video ID extraction fails:**
- Ask user to provide the full YouTube URL or video ID

**If video is restricted/private:**
- Inform user that the video cannot be accessed
- Ask them to check if the video is publicly available

## Quality Guidelines

- **Be concise:** Summary should be scannable in 30 seconds
- **Be accurate:** Don't add information not in the transcript
- **Be structured:** Use consistent formatting for easy reading
- **Be contextual:** Adjust detail level based on video length
  - Short videos (<5 min): Brief summary
  - Long videos (>30 min): More detailed breakdown

## Notes

- yt-dlp is a reliable tool that works on most YouTube videos
- Some videos may not have captions available
- Transcript quality depends on YouTube's auto-generated captions or manual captions
- Auto-generated captions may have errors, especially for technical terms or non-English content
