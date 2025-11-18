#!/usr/bin/env python3
"""
Test script to verify node disconnection detection.

This script will:
1. Register a test node with the Central Server
2. Send heartbeats for a while
3. Stop sending heartbeats
4. Verify that the Central Server detects disconnection and logs it
"""

import grpc
import sys
import time
from datetime import datetime

# Add the protocol directory to path
sys.path.insert(0, 'cyxwiz-protocol/python')

# Import generated protocol classes
from cyxwiz.protocol import node_pb2, node_pb2_grpc, common_pb2

def register_node(stub):
    """Register a test node with the Central Server"""
    request = node_pb2.RegisterNodeRequest(
        wallet_address="test_wallet_123",
        name="TestNode-DisconnectionTest",
        capabilities=node_pb2.NodeCapabilities(
            cpu_cores=8,
            ram_gb=32,
            gpu_model="NVIDIA RTX 3090",
            gpu_memory_gb=24,
            has_cuda=True,
            has_opencl=False
        ),
        endpoint=node_pb2.NodeEndpoint(
            ip_address="127.0.0.1",
            port=60000
        )
    )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Registering test node...")
    response = stub.RegisterNode(request)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Node registered: {response.node_id}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}]   Status: {common_pb2.StatusCode.Name(response.status)}")
    return response.node_id

def send_heartbeat(stub, node_id):
    """Send a heartbeat for the node"""
    request = node_pb2.HeartbeatRequest(
        node_id=node_id,
        status=node_pb2.NODE_STATUS_ONLINE,
        current_load=0.25,
        metrics=node_pb2.NodeMetrics(
            cpu_usage=25.0,
            memory_usage=50.0,
            gpu_usage=0.0,
            network_bandwidth=100.0
        )
    )

    response = stub.Heartbeat(request)
    return response.status == common_pb2.STATUS_SUCCESS

def main():
    # Connect to Central Server
    server_address = "localhost:50051"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Central Server at {server_address}...")

    channel = grpc.insecure_channel(server_address)
    stub = node_pb2_grpc.NodeServiceStub(channel)

    try:
        # Step 1: Register node
        node_id = register_node(stub)

        # Step 2: Send heartbeats for 20 seconds (keep node alive)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sending heartbeats for 20 seconds...")
        for i in range(4):
            if send_heartbeat(stub, node_id):
                print(f"[{datetime.now().strftime('%H:%M:%S')}]   Heartbeat #{i+1} sent successfully")
            time.sleep(5)

        # Step 3: Stop sending heartbeats and wait for disconnection detection
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠ STOPPING heartbeats...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting 40 seconds for Central Server to detect disconnection...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] (NodeMonitor checks every 10s, timeout is 30s)")
        print()
        print("Check the Central Server logs for:")
        print("  - WARN message: 'Node {id} ({name}) disconnected - no heartbeat for X seconds'")
        print("  - INFO message: 'Node {id} ({name}) marked as OFFLINE'")
        print()

        for i in range(8):
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   Waiting... ({(i+1)*5}/40 seconds)")
            time.sleep(5)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✓ Test complete!")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Check Central Server logs to verify disconnection was detected and logged.")

    except grpc.RpcError as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✗ gRPC Error: {e.code()} - {e.details()}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}]   Make sure Central Server is running on {server_address}")
        return 1
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✗ Error: {e}")
        return 1
    finally:
        channel.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
