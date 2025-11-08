import cv2
import mediapipe as mp
from collections import deque
import threading
import queue
import time

# ORIGINAL SOURCE: https://github.com/Elijah-cyber7/FreakDetector

# TODO: Make videos swappable

# === Settings ===

VIDEO_PATH_1 = "flight_tongue.mp4"
VIDEO_PATH_2 = "orca-tongue.mp4"
SHAKE_WINDOW = 15
SHAKE_THRESHOLD = 0.02
TONGUE_THRESHOLD = 0.01
MIN_MOUTH_OPEN = 0.02
TRIGGER_COOLDOWN = 60
SUSTAIN_FRAMES = 1
GESTURE_RUN_COUNT = 0

# === Setup MediaPipe ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

nose_positions = deque(maxlen=SHAKE_WINDOW)
gesture_frames = 0
cooldown = 0

# === Queue and flag for video frames ===
video_queue = queue.Queue(maxsize=1)
play_video_flag = threading.Event()

# === Function to read video frames in a separate thread ===
def video_reader():
    video_path = VIDEO_PATH_1   
    while True:   
        if video_path == VIDEO_PATH_1:
            video_path = VIDEO_PATH_2
        else:
            video_path = VIDEO_PATH_1  

        play_video_flag.wait()  # wait until flagged
        vid = cv2.VideoCapture(video_path)
        
        if not vid.isOpened():
            print("❌ Could not open video:", video_path)
            play_video_flag.clear()
            continue

        while vid.isOpened() and play_video_flag.is_set():
            ret, frame = vid.read()
            if not ret:
                break
            if not video_queue.full():
                video_queue.put(frame)
            time.sleep(0.01)  # reduce CPU usage

        vid.release()
        play_video_flag.clear()  # done playing

# Start video reader thread
threading.Thread(target=video_reader, daemon=True).start()

# === Gesture detection functions ===
def detect_tongue(landmarks):
    upper_lip = landmarks[13].y
    lower_lip = landmarks[14].y
    tongue_tip = landmarks[16].y
    mouth_height = lower_lip - upper_lip
    if mouth_height < MIN_MOUTH_OPEN:
        return False
    return (tongue_tip - lower_lip) > TONGUE_THRESHOLD

def detect_head_shake():
    if len(nose_positions) < SHAKE_WINDOW:
        return False
    motion = max(nose_positions) - min(nose_positions)
    return motion > SHAKE_THRESHOLD

# === Main loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)
    gesture_detected = False

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0].landmark
        nose_positions.append(face_landmarks[1].x)

        tongue_out = detect_tongue(face_landmarks)
        head_shake = detect_head_shake()

        if tongue_out and head_shake:
            gesture_detected = True

    # Count consecutive frames
    if gesture_detected:
        gesture_frames += 1
    else:
        gesture_frames = 0

    # Trigger video if sustained
    if gesture_frames >= SUSTAIN_FRAMES and cooldown == 0:
        GESTURE_RUN_COUNT+=1
        print("Gesture sustained for " + str(GESTURE_RUN_COUNT) + "! Starting video...")
        play_video_flag.set()
        cooldown = TRIGGER_COOLDOWN
        gesture_frames = 0

    # Cooldown counter
    if cooldown > 0:
        cooldown -= 1

    # Show webcam
    cv2.imshow("Freak Detector", frame)

    # Show video if available
    if not video_queue.empty():
        video_frame = video_queue.get()
        cv2.imshow("GETTING FREAKY", video_frame)
        cv2.moveWindow("Video Playback", 0, 0)
    elif not play_video_flag.is_set():
        # Video finished, close window
        cv2.destroyWindow("Video Playback")

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break
print("Gesture made " + str(GESTURE_RUN_COUNT) + " times")
cap.release()
cv2.destroyAllWindows()
