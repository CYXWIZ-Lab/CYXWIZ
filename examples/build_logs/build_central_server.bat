@echo off
REM Build script for CyxWiz Central Server with PROTOC set

set PROTOC=D:\Dev\CyxWiz_Claude\vcpkg\packages\protobuf_x64-windows\tools\protobuf\protoc.exe

cd /d D:\Dev\CyxWiz_Claude\cyxwiz-central-server

echo Building CyxWiz Central Server...
echo Using PROTOC: %PROTOC%

cargo build --release
