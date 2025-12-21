@echo off
REM Script to apply Phase 1 of gRPC enablement fixes
REM See GRPC_ENABLEMENT_GUIDE.md for full details

setlocal enabledelayedexpansion

echo ========================================
echo CyxWiz Central Server - gRPC Fix Script
echo ========================================
echo.
echo This script applies Phase 1 fixes from GRPC_ENABLEMENT_GUIDE.md
echo.
echo WARNING: This will modify source files. Make sure you have committed your changes.
echo.
pause

REM Step 1: Backup main.rs
echo [1/5] Backing up main.rs...
copy src\main.rs src\main.rs.backup >nul
echo checkmark main.rs backed up to src\main.rs.backup
echo.

REM Step 2: Note about pb.rs
echo [2/5] pb.rs module already created
echo The src\pb.rs file contains the protobuf module definition
echo.

REM Step 3: Manual editing required
echo [3/5] Manual editing required for main.rs
echo.
echo Please edit src\main.rs manually:
echo   1. Uncomment: mod api;
echo   2. Uncomment: mod blockchain;
echo   3. Uncomment: mod scheduler;
echo   4. Add after line 7: mod pb;
echo   5. Remove the commented pb module definition (lines 19-23)
echo   6. Add imports for gRPC services
echo.
echo See GRPC_ENABLEMENT_GUIDE.md Phase 1 for exact changes
echo.
pause

REM Step 4: Service files
echo [4/5] Service files need manual fixes
echo.
echo For EACH file in src\api\grpc\:
echo   - job_service.rs
echo   - node_service.rs
echo   - deployment_service.rs
echo   - terminal_service.rs
echo   - model_service.rs
echo.
echo Change:
echo   impl crate::pb::xxx_service_server::XxxServiceImpl {
echo To:
echo   impl XxxServiceImpl {
echo.
echo Also remove duplicate #[tonic::async_trait] annotations
echo.
pause

REM Step 5: Try building
echo [5/5] Ready to build
echo.
echo After making manual edits, try:
echo   cargo build --release 2^>^&1 ^| tee build.log
echo.
echo If errors occur, check build.log and GRPC_ENABLEMENT_GUIDE.md
echo.
echo To restore backup:
echo   copy src\main.rs.backup src\main.rs
echo.
echo ========================================
echo Next Steps
echo ========================================
echo.
echo 1. Make manual edits described above
echo 2. Run: cargo build --release
echo 3. Fix any remaining compilation errors
echo 4. Test with: cargo run -- --server
echo.
echo See GRPC_ENABLEMENT_GUIDE.md for complete procedure
echo.
pause
