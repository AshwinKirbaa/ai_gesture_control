@echo off
cd /d "%~dp0"
echo Starting AI Gesture Control using connected camera (GRANDSTREAM GUV3100)...
echo Face-gated: Gestures ONLY work when your face is detected!
echo.
echo Controls:
echo   - Move Index Finger: Mouse cursor
echo   - Pinch Thumb + Index: Left click
echo   - Pinch Thumb + Middle: Right click
echo   - Open Palm: Play/Pause media
echo   - Press 'e' in preview to enroll your face specifically
echo   - Press 'q' to quit, 'ESC' for emergency stop
echo.
.\.venv\Scripts\python.exe main_gesture.py
pause

