@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Dang cai dat/kiem tra thu vien can thiet...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Loi cai dat thu vien. Hay kiem tra Python va ket noi Internet.
    pause
    exit /b 1
)
echo.
echo App se tu mo tren trinh duyet. Neu khong tu mo, hay vao link hien trong cua so nay.
python app.py
pause
