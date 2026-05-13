@echo off
cd /d %~dp0
set FLASK_DEBUG=1
set FLASK_HOST=0.0.0.0
set FLASK_PORT=5000
start http://127.0.0.1:5000
python app.py
pause
