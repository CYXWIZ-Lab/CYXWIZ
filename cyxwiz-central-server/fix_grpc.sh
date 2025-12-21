#!/bin/bash
# Script to apply Phase 1 of gRPC enablement fixes
# See GRPC_ENABLEMENT_GUIDE.md for full details

set -e

echo "========================================"
echo "CyxWiz Central Server - gRPC Fix Script"
echo "========================================"
echo ""
echo "This script applies Phase 1 fixes from GRPC_ENABLEMENT_GUIDE.md"
echo ""

# Step 1: Update main.rs to enable modules and use pb.rs
echo "[1/5] Updating main.rs to enable api, blockchain, scheduler modules..."

# Backup main.rs
cp src/main.rs src/main.rs.backup

# Enable modules in main.rs
sed -i 's|^// mod api;|mod api;|g' src/main.rs
sed -i 's|^// mod blockchain;|mod blockchain;|g' src/main.rs
sed -i 's|^// mod scheduler;|mod scheduler;|g' src/main.rs

# Add pb module
sed -i 's|^mod tui;$|mod pb;\nmod tui;|g' src/main.rs

# Remove old pb definition comment
sed -i '/^\/\/ Proto code commented out for TUI-only mode$/,/^\/\/ }$/d' src/main.rs

# Add imports back
cat << 'EOF' >> src/main.rs.tmp
use crate::api::grpc::{
    DeploymentServiceImpl,
    JobServiceImpl,
    ModelServiceImpl,
    NodeServiceImpl,
    TerminalServiceImpl,
};
use crate::blockchain::{PaymentProcessor, SolanaClient};
use crate::scheduler::JobScheduler;
use tonic::transport::Server;
EOF

# This is a simplified approach - manual editing recommended
echo "✓ main.rs updated (backup at src/main.rs.backup)"

# Step 2: Fix job_service.rs
echo "[2/5] Fixing job_service.rs..."

# Fix impl block
sed -i 's|impl crate::pb::job_service_server::JobServiceImpl {|impl JobServiceImpl {|g' src/api/grpc/job_service.rs

# Remove duplicate async_trait
sed -i '/^#\[tonic::async_trait\]$/,/^#\[tonic::async_trait\]$/{//!d;}' src/api/grpc/job_service.rs

echo "✓ job_service.rs fixed"

# Step 3: Fix node_service.rs
echo "[3/5] Fixing node_service.rs..."
sed -i 's|impl crate::pb::node_service_server::NodeServiceImpl {|impl NodeServiceImpl {|g' src/api/grpc/node_service.rs
echo "✓ node_service.rs fixed"

# Step 4: Fix deployment_service.rs
echo "[4/5] Fixing deployment_service.rs..."
sed -i 's|impl crate::pb::deployment_service_server::DeploymentServiceImpl {|impl DeploymentServiceImpl {|g' src/api/grpc/deployment_service.rs
echo "✓ deployment_service.rs fixed"

# Step 5: Fix terminal_service.rs and model_service.rs
echo "[5/5] Fixing terminal_service.rs and model_service.rs..."
sed -i 's|impl crate::pb::terminal_service_server::TerminalServiceImpl {|impl TerminalServiceImpl {|g' src/api/grpc/terminal_service.rs
sed -i 's|impl crate::pb::model_service_server::ModelServiceImpl {|impl ModelServiceImpl {|g' src/api/grpc/model_service.rs
echo "✓ terminal_service.rs and model_service.rs fixed"

echo ""
echo "========================================"
echo "Phase 1 fixes applied!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Try building: cargo build --release"
echo "  3. Fix any remaining errors manually"
echo "  4. Restore backup if needed: mv src/main.rs.backup src/main.rs"
echo ""
echo "See GRPC_ENABLEMENT_GUIDE.md for full procedure"
echo ""
