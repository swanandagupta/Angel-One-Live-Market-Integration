@echo off
echo =========================================================================
echo  AI/ML Stock Market Screening and Analysis System — PyInstaller Build
echo =========================================================================

echo 1. Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo 2. Generating sample dataset and training ML model...
python -m ml.train

echo 3. Compiling Windows Standalone Executable...
python -m PyInstaller --noconfirm run_app.spec

echo.
echo =========================================================================
echo  BUILD COMPLETE!
echo  Executable Location: dist\run_app\run_app.exe
echo =========================================================================
pause
