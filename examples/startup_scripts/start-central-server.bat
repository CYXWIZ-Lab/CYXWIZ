@echo off
REM Start CyxWiz Central Server
echo Starting CyxWiz Central Server...
echo.

REM Check if already running
netstat -ano | findstr ":50051.*LISTENING" >nul
if %errorlevel% == 0 (
    echo ERROR: Central Server is already running on port 50051
    echo Run stop-all.bat to stop existing services
    pause
    exit /b 1
)

REM Start Central Server
cd cyxwiz-central-server
echo Starting on ports 50051 (gRPC) and 8080 (REST)...
echo Press Ctrl+C to stop
echo.
cargo run --release
