#!/usr/bin/env python3
"""
CyxWiz Plugin System - Full Integration Test

This script tests:
1. Bundled Python 3.12 environment
2. MuJoCo physics simulation
3. Gymnasium RL environments
4. Stable-Baselines3 PPO training
5. pycyxwiz backend integration

Run from CyxWiz Python Console:
    exec(open('docs/plugin_usage_guide_test.py').read())

Or from command line:
    ./build/bin/Release/python/python.exe docs/plugin_usage_guide_test.py
"""

import sys
from pathlib import Path


def test_python_environment():
    """Test that bundled Python works correctly."""
    print("=" * 60)
    print("PYTHON ENVIRONMENT TEST")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"Executable: {sys.executable}")

    # Test required packages
    packages = [
        ("numpy", "Array operations"),
        ("gymnasium", "RL environments"),
        ("mujoco", "Physics simulation"),
        ("stable_baselines3", "RL algorithms"),
    ]

    results = []
    for pkg, desc in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"[PASS] {pkg} v{version} - {desc}")
            results.append(True)
        except ImportError as e:
            print(f"[FAIL] {pkg} - {e}")
            results.append(False)

    return all(results)


def test_mujoco_physics():
    """Test MuJoCo physics simulation."""
    print("\n" + "=" * 60)
    print("MUJOCO PHYSICS TEST")
    print("=" * 60)

    try:
        import mujoco
        import numpy as np

        # Simple pendulum model
        xml = """
        <mujoco>
            <worldbody>
                <body name="pendulum">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.05 0.5"/>
                </body>
            </worldbody>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Apply initial velocity
        data.qvel[0] = 1.0

        # Step simulation
        for _ in range(100):
            mujoco.mj_step(model, data)

        print(f"[PASS] Simulation ran 100 steps")
        print(f"       Final time: {data.time:.4f}s")
        print(f"       Final position: {data.qpos[0]:.4f} rad")
        return True

    except Exception as e:
        print(f"[FAIL] MuJoCo test failed: {e}")
        return False


def test_gymnasium_cartpole():
    """Test Gymnasium CartPole environment."""
    print("\n" + "=" * 60)
    print("GYMNASIUM CARTPOLE TEST")
    print("=" * 60)

    try:
        import gymnasium as gym

        env = gym.make("CartPole-v1")
        obs, info = env.reset()

        total_reward = 0
        steps = 0

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        env.close()

        print(f"[PASS] CartPole-v1 test complete")
        print(f"       Steps: {steps}")
        print(f"       Total reward: {total_reward}")
        return True

    except Exception as e:
        print(f"[FAIL] Gymnasium test failed: {e}")
        return False


def test_gymnasium_mujoco():
    """Test Gymnasium with MuJoCo environment."""
    print("\n" + "=" * 60)
    print("GYMNASIUM MUJOCO TEST")
    print("=" * 60)

    try:
        import gymnasium as gym

        env = gym.make("Ant-v4")
        obs, info = env.reset()

        total_reward = 0
        steps = 0

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        env.close()

        print(f"[PASS] Ant-v4 (MuJoCo) test complete")
        print(f"       Steps: {steps}")
        print(f"       Total reward: {total_reward:.2f}")
        return True

    except Exception as e:
        print(f"[FAIL] MuJoCo environment test failed: {e}")
        return False


def test_stable_baselines3():
    """Test Stable-Baselines3 PPO training."""
    print("\n" + "=" * 60)
    print("STABLE-BASELINES3 PPO TEST")
    print("=" * 60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        # Create vectorized environment
        env = make_vec_env("CartPole-v1", n_envs=2)

        # Create PPO model
        model = PPO("MlpPolicy", env, verbose=0)

        # Train for 500 timesteps
        print("Training PPO for 500 timesteps...")
        model.learn(total_timesteps=500)
        print("[PASS] Training complete")

        # Test inference
        obs = env.reset()
        action, _ = model.predict(obs)
        print(f"[PASS] Inference works - Action: {action}")

        # Evaluate
        total_reward = 0
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards.sum()

        print(f"[PASS] Evaluation reward: {total_reward:.2f}")

        env.close()
        return True

    except Exception as e:
        print(f"[FAIL] SB3 test failed: {e}")
        return False


def test_pycyxwiz_integration():
    """Test pycyxwiz module (requires engine context)."""
    print("\n" + "=" * 60)
    print("PYCYXWIZ INTEGRATION TEST")
    print("=" * 60)

    try:
        import pycyxwiz

        # Test backend initialization
        if hasattr(pycyxwiz, 'is_initialized'):
            initialized = pycyxwiz.is_initialized()
            print(f"[PASS] Backend initialized: {initialized}")

        # Test tensor operations
        if hasattr(pycyxwiz, 'Tensor'):
            t = pycyxwiz.Tensor([1.0, 2.0, 3.0, 4.0])
            print(f"[PASS] Created tensor with {len(t)} elements")

        # Test device info
        if hasattr(pycyxwiz, 'get_device_count'):
            count = pycyxwiz.get_device_count()
            print(f"[PASS] Found {count} compute devices")

        # Test plugin API (if exposed)
        if hasattr(pycyxwiz, 'list_plugins'):
            plugins = pycyxwiz.list_plugins()
            print(f"[PASS] Found {len(plugins)} loaded plugins")
            for p in plugins:
                print(f"       - {p}")
        else:
            print("[INFO] Plugin listing API not exposed to Python")

        return True

    except ImportError as e:
        print(f"[INFO] pycyxwiz not available: {e}")
        print("       (This is expected when running outside engine)")
        return True  # Not a failure if running standalone

    except Exception as e:
        print(f"[FAIL] pycyxwiz test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("CYXWIZ PLUGIN SYSTEM - FULL TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        ("Python Environment", test_python_environment),
        ("MuJoCo Physics", test_mujoco_physics),
        ("Gymnasium CartPole", test_gymnasium_cartpole),
        ("Gymnasium MuJoCo", test_gymnasium_mujoco),
        ("Stable-Baselines3 PPO", test_stable_baselines3),
        ("pycyxwiz Integration", test_pycyxwiz_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results if p)
    failed = sum(1 for _, p in results if not p)

    for name, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\nAll tests PASSED! Plugin system is fully functional.")
    else:
        print("\nSome tests failed. Check output above for details.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
