===========================================================
                Facial Gesture Video Trigger
===========================================================

This program uses your webcam and MediaPipe’s Face Mesh model
to detect a specific facial gesture — sticking out your tongue 
while shaking your head — and plays a video automatically 
when the gesture is sustained for a short duration.

-----------------------------------------------------------
REQUIREMENTS
-----------------------------------------------------------
1. Python 3.8+
2. OpenCV
3. MediaPipe

Install dependencies with:
    pip install opencv-python mediapipe

-----------------------------------------------------------
HOW IT WORKS
-----------------------------------------------------------
1. The webcam feed runs continuously.
2. MediaPipe detects facial landmarks (nose, lips, tongue).
3. The script tracks:
   - Nose position changes (head shake motion)
   - Mouth openness and tongue position
4. When both the tongue-out and head-shake gestures are
   sustained for a set number of frames (default = 7),
   a local video (orca-tongue.mp4) plays in a new window.

After the video finishes, it closes automatically.

-----------------------------------------------------------
CONFIGURATION
-----------------------------------------------------------
You can edit the parameters at the top of the file:

    VIDEO_PATH       — path to the video file to play  
    SHAKE_WINDOW     — number of frames to track head motion  
    SHAKE_THRESHOLD  — sensitivity for head movement  
    TONGUE_THRESHOLD — how far the tongue must extend  
    MIN_MOUTH_OPEN   — minimum mouth opening to consider  
    TRIGGER_COOLDOWN — frames to wait before retriggering  
    SUSTAIN_FRAMES   — gesture must last this many frames  

-----------------------------------------------------------
LOGIC OVERVIEW
-----------------------------------------------------------
- Facial landmarks are read from MediaPipe in each frame.
- The horizontal position of the nose tip is stored to detect
  shaking motion.
- Lip and tongue positions determine whether the mouth is open
  and the tongue is extended.
- When both are true for several frames in a row, the video
  playback thread activates.
- The playback happens in a separate thread so the webcam feed
  remains responsive.

-----------------------------------------------------------
OUTPUT WINDOWS
-----------------------------------------------------------
1. "Facial Gesture Detection" — Live webcam feed.
2. "Video Playback" — Plays the triggered video.

Both windows can be closed by pressing the ESC key.

-----------------------------------------------------------
AUTHOR
-----------------------------------------------------------
Created by Andrew Allen  
Instagram: ElijahCyber 
LinkedIn: https://www.linkedin.com/in/andrew-allen-655499190/
Last updated: October 2025  

-----------------------------------------------------------
NOTES
-----------------------------------------------------------
- Make sure your webcam is connected and accessible.  
- Ensure your lighting is adequate for MediaPipe to detect
  facial landmarks accurately.  
- Replace the default video file with any .mp4 you prefer.

===========================================================
