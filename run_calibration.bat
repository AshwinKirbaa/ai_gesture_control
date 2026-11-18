@echo off
cd /d "%~dp0"
echo Starting AI Gesture Control in CALIBRATION MODE (Preview only, no mouse clicks/movement)...
echo Face-gated: Gestures ONLY show when your face is detected!
echo.
echo Press 'e' in preview to enroll your face specifically.
echo Press 'q' to quit, 'ESC' for emergency stop.
echo.
.\.venv\Scripts\python.exe main_gesture.py --calibrate
pause

