@echo off
cd /d "%~dp0"
echo Registering Owner Face using connected camera...
echo.
echo Please look at the camera and tilt/turn your head slightly.
echo Press 'q' to stop when enough samples are captured.
echo.
.\.venv\Scripts\python.exe register_face.py
pause

