@echo off
REM Quick runner for training examples with proper DLL path setup

echo ===================================================================
echo CyxWiz Training Example Runner
echo ===================================================================
echo.

REM Change to test_clean directory where all DLLs are present
cd build\test_clean

REM Copy the training script if not already there
if not exist "test_training_xor_visualized.py" (
    echo Copying training example to test_clean...
    copy /Y "..\..\test_training_xor_visualized.py" .
)

REM Update the script to use local paths
python -c "import sys; open('test_xor_local.py', 'w').write(open('test_training_xor_visualized.py').read().replace('D:/Dev/CyxWiz_Claude/build/windows-release/bin/Release', '.').replace('D:/Dev/CyxWiz_Claude/build/windows-release/lib/Release', '.'))"

echo Running XOR training with visualization...
echo.
python test_xor_local.py

echo.
echo ===================================================================
echo Training complete! Check the output above.
echo ===================================================================
pause
