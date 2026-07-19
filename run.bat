@echo off
setlocal

REM cd to repo root (directory containing this script)
cd /d "%~dp0"

REM Find python
set PYTHON=
where python >nul 2>nul
if %ERRORLEVEL%==0 set PYTHON=python
if not defined PYTHON (
    where py >nul 2>nul
    if %ERRORLEVEL%==0 set PYTHON=py
)
if not defined PYTHON (
    echo Error: python / py not found in PATH
    pause
    exit /b 1
)

echo == pytorch-weights-only-state-dict-lab ==
%PYTHON% --version 2>&1
%PYTHON% -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 echo PyTorch: not installed (torch-dependent cases will framework_skip -- this is normal)
echo.

echo --- running lab ---
%PYTHON% run_lab.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Lab run failed with error %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo --- running tests ---
%PYTHON% -m unittest -v
set TEST_EXIT=%ERRORLEVEL%
echo.

if %TEST_EXIT% NEQ 0 (
    echo Tests failed with exit code %TEST_EXIT%
    pause
    exit /b %TEST_EXIT%
)

echo Done. See observations.json / observations.csv / RESULTS.md
pause
