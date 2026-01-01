# 🏋️ AI Fitness Coach – Pose-Based Posture Correction System

## Overview
This project is a **real-time exercise posture correction system** built using
computer vision and pose estimation.

It analyzes body posture from a **live webcam feed** using **MediaPipe Pose** and
provides **real-time visual and voice feedback** during exercises.

The system is **rule-based and deterministic** — no training, no datasets, and no
machine learning model fitting.

The goal is **form correctness, symmetry, and injury prevention**, not just rep counting.

---

## Features
- Real-time pose detection using webcam
- Supports multiple exercises:
  - Push-up
  - Squat
  - Lunge
  - Plank
- Automatic posture calibration per user
- Joint-angle–based posture validation
- Symmetry checks (left vs right limbs)
- Real-time repetition counting (non-plank exercises)
- Time tracking for plank exercise
- Voice feedback using text-to-speech
- On-screen posture and performance metrics
- Workout summary with stability score

---

## How It Works

### 1. Pose Detection
- Webcam frames are captured using **OpenCV**
- **MediaPipe Pose** extracts 3D body landmarks per frame

---

### 2. Angle Computation
- Joint angles are calculated using vector geometry
- Examples:
  - Elbow angle (push-ups)
  - Knee angle (squats and lunges)
  - Hip and torso alignment (plank)

---

### 3. Auto Calibration
- User performs **one correct repetition**
- System learns:
  - Minimum angle
  - Maximum angle
  - Depth threshold
- Calibration is user-specific and automatic

---

### 4. Posture Validation
- Exercise-specific rules are applied:
  - Minimum depth
  - Limb symmetry
  - Torso alignment
- Incorrect posture triggers warnings

---

### 5. Repetition & Time Tracking
- Valid reps are counted only when posture rules are met
- Plank duration is tracked in seconds

---

### 6. Feedback System
- On-screen text feedback
- Real-time **voice feedback** using text-to-speech
- Cooldown prevents repeated audio spam

---

## Why This Is Not a Machine Learning Model
- No training data
- No classification or prediction model
- No labels or accuracy metrics

This is a **rule-based AI system**, which is:
- More interpretable
- More reliable for real-time correction
- Easier to debug and validate

---

## Tech Stack

### Core
- **Python**

### Frontend
- **Streamlit** – UI and app control

### Computer Vision
- **MediaPipe Pose** – Body landmark detection
- **OpenCV** – Video capture and rendering

### Math & Processing
- **NumPy** – Vector and angle calculations

### Audio
- **pyttsx3** – Offline text-to-speech feedback
- **threading** – Non-blocking audio output

