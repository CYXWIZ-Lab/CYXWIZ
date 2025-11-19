"""
Configuration for Python Build Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory (parent of python_build_agent)
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_AGENT_ROOT = Path(__file__).parent

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model selection
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"  # Latest Claude model
OPENAI_MODEL = "gpt-4o"

# Agent behavior
MAX_ITERATIONS = 10  # Maximum attempts to fix build errors
MAX_TOOL_CALLS = 50  # Maximum tool calls per iteration
TIMEOUT_SECONDS = 300  # Timeout for shell commands

# Build configuration
BUILD_PRESET = "windows-release"  # Default CMake preset
BUILD_DIR = PROJECT_ROOT / "build"
VCPKG_ROOT = PROJECT_ROOT / "vcpkg"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = BUILD_AGENT_ROOT / "build_agent.log"

# Error patterns to recognize
COMMON_ERROR_PATTERNS = [
    r"CMake Error",
    r"fatal error",
    r"error C\d+:",
    r"undefined reference",
    r"cannot find",
    r"No such file or directory",
    r"permission denied",
]
