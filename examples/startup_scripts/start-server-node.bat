@echo off
REM Start CyxWiz Server Node
echo Starting CyxWiz Server Node...
echo.

REM Check if already running
netstat -ano | findstr ":50052.*LISTENING" >nul
if %errorlevel% == 0 (
    echo ERROR: Server Node is already running on port 50052
    echo Run stop-all.bat to stop existing services
    pause
    exit /b 1
)

REM Check if Central Server is running
netstat -ano | findstr ":50051.*LISTENING" >nul
if %errorlevel% neq 0 (
    echo WARNING: Central Server is not running on port 50051
    echo The Server Node will try to connect but may fail
    echo Start the Central Server first with start-central-server.bat
    echo.
    echo Press any key to continue anyway, or Ctrl+C to abort...
    pause >nul
)

REM Start Server Node
echo Starting on ports 50052 (Deployment), 50053 (Terminal), 50054 (NodeService)...
echo Press Ctrl+C to stop
echo.
.\build\windows-release\bin\Release\cyxwiz-server-node.exe
