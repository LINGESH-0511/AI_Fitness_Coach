import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading

# ===================== TEXT TO SPEECH =====================
def speak(text):
    def _run(msg):
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(msg)
        engine.runAndWait()
    threading.Thread(target=_run, args=(text,), daemon=True).start()

# ===================== MEDIAPIPE =====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===================== UTILS =====================
def angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

def symmetry_diff(left_angle, right_angle):
    return abs(left_angle - right_angle)

# ===================== STREAMLIT UI =====================
st.set_page_config("AI Fitness Coach – Pose-Based Prototype", layout="wide")
st.title("🏋️ AI Fitness Coach – Pose-Based Prototype")

exercise = st.selectbox(
    "Select Exercise",
    ["Push-up", "Squat", "Lunge", "Plank"]
)

# ===================== SESSION STATE =====================
if "running" not in st.session_state:
    st.session_state.update({
        "running": False,
        "calibrated": False,
        "counter": 0,
        "state": "UP",
        "feedback_log": {},
        "calib_angles": [],
        "min_angle": None,
        "max_angle": None,
        "threshold": None,
        "plank_start": None,
        "plank_time": 0,
        "bad_reps": 0,
        "depth_total": 0
    })

# ===================== BUTTONS =====================
col1, col2 = st.columns(2)
with col1:
    start = st.button("Start Workout")
with col2:
    stop = st.button("Stop Workout")

if start:
    st.session_state.update({
        "running": True,
        "calibrated": False,
        "counter": 0,
        "state": "UP",
        "feedback_log": {"shallow": 0, "bad_posture": 0},
        "calib_angles": [],
        "min_angle": None,
        "max_angle": None,
        "threshold": None,
        "plank_start": None,
        "plank_time": 0,
        "bad_reps": 0,
        "depth_total": 0
    })

if stop:
    st.session_state.running = False

frame_box = st.empty()
status_box = st.empty()

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
last_voice = 0
VOICE_COOLDOWN = 2

# ===================== FRAME PROCESSING =====================
def process_frame(frame):
    global last_voice
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    now = time.time()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ====== MULTI-ANGLE CALCULATIONS ======
        angles = {}
        if exercise == "Push-up":
            angles["elbow_left"] = angle_3d(lm[11], lm[13], lm[15])
            angles["elbow_right"] = angle_3d(lm[12], lm[14], lm[16])
            angles["shoulder"] = angle_3d(lm[23], lm[11], lm[13])
            angles["hip"] = angle_3d(lm[11], lm[23], lm[25])
        elif exercise == "Squat":
            angles["knee_left"] = angle_3d(lm[23], lm[25], lm[27])
            angles["knee_right"] = angle_3d(lm[24], lm[26], lm[28])
            angles["hip"] = angle_3d(lm[11], lm[23], lm[25])
            angles["torso"] = angle_3d(lm[11], lm[23], lm[25])
        elif exercise == "Lunge":
            angles["knee_front"] = angle_3d(lm[24], lm[26], lm[28])
            angles["knee_back"] = angle_3d(lm[23], lm[25], lm[27])
            angles["hip_front"] = angle_3d(lm[11], lm[23], lm[25])
            angles["torso"] = angle_3d(lm[11], lm[23], lm[25])
        elif exercise == "Plank":
            angles["shoulder_left"] = angle_3d(lm[11], lm[13], lm[15])
            angles["shoulder_right"] = angle_3d(lm[12], lm[14], lm[16])
            angles["hip_left"] = angle_3d(lm[23], lm[25], lm[27])
            angles["hip_right"] = angle_3d(lm[24], lm[26], lm[28])
            angles["torso"] = angle_3d(lm[11], lm[23], lm[27])

        # ===================== CALIBRATION =====================
        avg_angle = np.mean(list(angles.values()))
        if not st.session_state.calibrated:
            st.session_state.calib_angles.append(avg_angle)
            status_box.info("Calibrating... Perform ONE proper rep")
            if len(st.session_state.calib_angles) >= 60:
                st.session_state.min_angle = min(st.session_state.calib_angles)
                st.session_state.max_angle = max(st.session_state.calib_angles)
                st.session_state.threshold = st.session_state.min_angle + 0.15 * (
                    st.session_state.max_angle - st.session_state.min_angle
                )
                st.session_state.calibrated = True
                speak("Calibration complete. Start your workout.")
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            return frame

        threshold = st.session_state.threshold

        # ===================== EXERCISE LOGIC =====================
        rep_complete = False
        posture_warning = None

        if exercise != "Plank":
            if avg_angle > st.session_state.max_angle - 5:
                st.session_state.state = "UP"
            elif avg_angle < threshold and st.session_state.state == "UP":
                st.session_state.state = "DOWN"
                st.session_state.counter += 1
                rep_complete = True
                st.session_state.depth_total += st.session_state.max_angle - avg_angle
                speak(f"Rep {st.session_state.counter}")

            # Symmetry check
            if "elbow_left" in angles and "elbow_right" in angles:
                diff = symmetry_diff(angles["elbow_left"], angles["elbow_right"])
                if diff > 15 and now - last_voice > VOICE_COOLDOWN:
                    posture_warning = "Keep both arms symmetrical"
                    last_voice = now
            if "knee_left" in angles and "knee_right" in angles:
                diff = symmetry_diff(angles["knee_left"], angles["knee_right"])
                if diff > 15 and now - last_voice > VOICE_COOLDOWN:
                    posture_warning = "Keep legs symmetrical"
                    last_voice = now

            # Predictive feedback
            if avg_angle > threshold + 20 and now - last_voice > VOICE_COOLDOWN:
                posture_warning = "Lower slowly and maintain alignment"
                last_voice = now

            if posture_warning:
                speak(posture_warning)
                st.session_state.feedback_log["bad_posture"] += 1

        else:  # Plank
            torso_angle = angles["torso"]
            if 170 <= torso_angle <= 185:
                if st.session_state.plank_start is None:
                    st.session_state.plank_start = time.time()
                st.session_state.plank_time = int(time.time() - st.session_state.plank_start)
            else:
                if now - last_voice > VOICE_COOLDOWN:
                    speak("Keep your torso straight")
                    last_voice = now
                st.session_state.plank_start = None
                st.session_state.feedback_log["bad_posture"] += 1

            # Left/right hip symmetry
            if "hip_left" in angles and "hip_right" in angles:
                diff = symmetry_diff(angles["hip_left"], angles["hip_right"])
                if diff > 10 and now - last_voice > VOICE_COOLDOWN:
                    speak("Keep hips level")
                    last_voice = now
                    st.session_state.feedback_log["bad_posture"] += 1

        # ===================== DRAW LANDMARKS =====================
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Reps: {st.session_state.counter}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if not exercise == "Plank":
            cv2.putText(frame, f"Avg Angle: {int(avg_angle)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, f"Plank Time: {st.session_state.plank_time}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

# ===================== STREAMLIT LOOP =====================
if st.session_state.running:
    while True:
        ret, frame = cap.read()
        if not ret or not st.session_state.running:
            break
        processed_frame = process_frame(frame)
        frame_box.image(processed_frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# ===================== SUMMARY =====================
if not st.session_state.running and st.session_state.calibrated:
    st.subheader("🏁 Workout Summary")
    if exercise == "Plank":
        st.write(f"**Time Held:** {st.session_state.plank_time} seconds")
    else:
        st.write(f"**Valid Reps:** {st.session_state.counter}")
        if st.session_state.counter > 0:
            avg_depth = st.session_state.depth_total / st.session_state.counter
            st.write(f"**Avg Depth:** {avg_depth:.1f} degrees")
            bad_percentage = (st.session_state.feedback_log["bad_posture"] / st.session_state.counter) * 100
            st.write(f"**Bad Posture %:** {bad_percentage:.1f}%")
            stability_score = max(0, 100 - bad_percentage)
            st.write(f"**Stability Score:** {stability_score:.1f}/100")

    st.write("**Feedback Messages:**")
    feedback_messages = {
        "Push-up": ["Go lower", "Keep body straight", "Elbows in", "Maintain symmetry"],
        "Squat": ["Go lower", "Push knees out", "Chest up", "Keep torso upright", "Legs aligned"],
        "Lunge": ["Lower back knee", "Stay upright", "Keep front knee aligned", "Maintain balance"],
        "Plank": ["Keep your torso straight", "Hips up/down", "Hips level"]
    }
    for msg in feedback_messages[exercise]:
        st.write(f"- {msg}")
