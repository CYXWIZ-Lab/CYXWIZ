"""
Command Executor Tool - Execute shell commands and capture output
"""
import subprocess
import os
from pathlib import Path
from typing import Dict, Any
from config import PROJECT_ROOT, TIMEOUT_SECONDS


class CommandExecutor:
    """Execute shell commands safely"""

    def __init__(self, working_dir: Path = PROJECT_ROOT):
        self.working_dir = working_dir

    def execute(
        self,
        command: str,
        timeout: int = TIMEOUT_SECONDS,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a shell command

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dict with 'success', 'stdout', 'stderr', 'exit_code'
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.working_dir),
                timeout=timeout,
                capture_output=capture_output,
                text=True,
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Error executing command: {str(e)}",
            }

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for LLM function calling"""
        from utils.llm_client import create_tool_definition

        return create_tool_definition(
            name="execute_command",
            description="Execute a shell command in the project directory. Use this to run cmake, make, build commands, or check system state.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'cmake --version', 'cmake --preset windows-release')",
                },
            },
        )

    def handle_tool_call(self, arguments: Dict[str, Any]) -> str:
        """Handle tool call from LLM"""
        command = arguments.get("command", "")
        result = self.execute(command)

        if result["success"]:
            return f"Command executed successfully:\n{result['stdout']}"
        else:
            return f"Command failed with exit code {result['exit_code']}:\nSTDERR: {result['stderr']}\nSTDOUT: {result['stdout']}"


# Singleton instance
command_executor = CommandExecutor()
