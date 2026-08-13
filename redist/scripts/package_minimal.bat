@echo off
setlocal

if defined CYXWIZ_PACKAGING_PYTHON goto custom_python
where py >nul 2>nul
if errorlevel 1 goto system_python
py -3 "%~dp0package_release.py" minimal %*
exit /b %errorlevel%

:system_python
python "%~dp0package_release.py" minimal %*
exit /b %errorlevel%

:custom_python
"%CYXWIZ_PACKAGING_PYTHON%" "%~dp0package_release.py" minimal %*
exit /b %errorlevel%
