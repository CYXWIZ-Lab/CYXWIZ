@echo off
REM Stop all CyxWiz services
echo Stopping all CyxWiz services...
echo.

REM Find and kill Central Server (ports 50051, 8080)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":50051.*LISTENING"') do (
    echo Stopping Central Server (PID %%a)...
    taskkill /F /PID %%a 2>nul
)

REM Find and kill Server Node (ports 50052, 50053, 50054)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":50052.*LISTENING"') do (
    echo Stopping Server Node (PID %%a)...
    taskkill /F /PID %%a 2>nul
)

echo.
echo All services stopped.
echo.
pause
