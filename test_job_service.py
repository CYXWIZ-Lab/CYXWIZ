#!/usr/bin/env python3
"""
Test script to verify JobService is working.

This script will:
1. Connect to the Central Server
2. Submit a test training job
3. Query the job status
4. Display the results
"""

import grpc
import sys
from datetime import datetime

# Add the protocol directory to path
sys.path.insert(0, 'cyxwiz-protocol/python')

# Import generated protocol classes
try:
    from cyxwiz.protocol import job_pb2, job_pb2_grpc, common_pb2
    print("✓ Protocol imports successful")
except ImportError as e:
    print(f"✗ Failed to import protocol: {e}")
    print("  Make sure to generate Python bindings first:")
    print("  cd cyxwiz-protocol && python -m grpc_tools.protoc ...")
    sys.exit(1)

def test_submit_job(stub):
    """Test job submission"""
    print("\n[TEST 1] Submitting a training job...")

    # Create a job configuration
    config = job_pb2.JobConfig(
        job_type=job_pb2.JOB_TYPE_TRAINING,
        model_definition='{"layers": [{"type": "dense", "units": 128}]}',
        dataset_uri="s3://datasets/mnist.tar.gz",
        batch_size=32,
        epochs=10,
        required_device=common_pb2.DEVICE_CUDA,
        estimated_memory=2 * 1024 * 1024 * 1024,  # 2GB
        estimated_duration=3600,  # 1 hour
        payment_address="test_wallet_address_123",
    )
    # Add hyperparameters
    config.hyperparameters["learning_rate"] = "0.001"
    config.hyperparameters["optimizer"] = "adam"

    request = job_pb2.SubmitJobRequest(config=config)

    try:
        response = stub.SubmitJob(request)

        if response.status == common_pb2.STATUS_SUCCESS:
            print(f"  ✓ Job submitted successfully!")
            print(f"    Job ID: {response.job_id}")
            print(f"    Status: {common_pb2.StatusCode.Name(response.status)}")
            return response.job_id
        else:
            print(f"  ✗ Job submission failed")
            if response.error:
                print(f"    Error: {response.error.message}")
            return None

    except grpc.RpcError as e:
        print(f"  ✗ RPC Error: {e.code()} - {e.details()}")
        return None

def test_get_job_status(stub, job_id):
    """Test job status query"""
    print(f"\n[TEST 2] Querying job status for {job_id}...")

    request = job_pb2.GetJobStatusRequest(job_id=job_id)

    try:
        response = stub.GetJobStatus(request)

        if response.status:
            print(f"  ✓ Job status retrieved!")
            print(f"    Job ID: {response.status.job_id}")
            print(f"    Status: {common_pb2.StatusCode.Name(response.status.status)}")
            print(f"    Progress: {response.status.progress * 100:.1f}%")
            if response.status.current_node_id:
                print(f"    Assigned Node: {response.status.current_node_id}")
            if response.status.error:
                print(f"    Error: {response.status.error.message}")
            return True
        else:
            print(f"  ✗ No status returned")
            return False

    except grpc.RpcError as e:
        print(f"  ✗ RPC Error: {e.code()} - {e.details()}")
        return False

def test_cancel_job(stub, job_id):
    """Test job cancellation"""
    print(f"\n[TEST 3] Cancelling job {job_id}...")

    request = job_pb2.CancelJobRequest(
        job_id=job_id,
        reason="Test cancellation"
    )

    try:
        response = stub.CancelJob(request)

        status_name = common_pb2.StatusCode.Name(response.status)
        print(f"  ✓ Cancel request processed")
        print(f"    Status: {status_name}")
        print(f"    Refund Issued: {response.refund_issued}")

        if response.error:
            print(f"    Note: {response.error.message}")

        return True

    except grpc.RpcError as e:
        print(f"  ✗ RPC Error: {e.code()} - {e.details()}")
        return False

def test_unimplemented_rpcs(stub):
    """Test unimplemented RPCs"""
    print(f"\n[TEST 4] Testing unimplemented RPCs...")

    # Test StreamJobUpdates (should return UNIMPLEMENTED)
    print("  Testing StreamJobUpdates...")
    try:
        request = job_pb2.GetJobStatusRequest(job_id="test-job-id")
        stream = stub.StreamJobUpdates(request)
        for update in stream:
            print(f"    Received update: {update}")
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            print(f"    ✓ Correctly returns UNIMPLEMENTED")
        else:
            print(f"    ✗ Unexpected error: {e.code()}")

    # Test ListJobs (should return UNIMPLEMENTED)
    print("  Testing ListJobs...")
    try:
        request = job_pb2.ListJobsRequest(user_id="test_user")
        response = stub.ListJobs(request)
        print(f"    Jobs: {len(response.jobs)}")
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            print(f"    ✓ Correctly returns UNIMPLEMENTED")
        else:
            print(f"    ✗ Unexpected error: {e.code()}")

def main():
    # Connect to Central Server
    server_address = "localhost:50051"
    print(f"Connecting to Central Server at {server_address}...")

    try:
        channel = grpc.insecure_channel(server_address)
        stub = job_pb2_grpc.JobServiceStub(channel)
        print("✓ Connected to JobService\n")
        print("=" * 60)

        # Run tests
        job_id = test_submit_job(stub)

        if job_id:
            test_get_job_status(stub, job_id)
            test_cancel_job(stub, job_id)

        test_unimplemented_rpcs(stub)

        print("\n" + "=" * 60)
        print("✓ JobService tests completed!")

        channel.close()
        return 0

    except grpc.RpcError as e:
        print(f"\n✗ Connection failed: {e.code()} - {e.details()}")
        print(f"  Make sure Central Server is running on {server_address}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
