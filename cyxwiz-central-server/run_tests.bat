@echo off
cd /d D:\Dev\CyxWiz_Claude\cyxwiz-central-server
echo Running JobStatusService integration tests...
cargo test --test job_status_service_tests
