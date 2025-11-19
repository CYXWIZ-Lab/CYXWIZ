"""
File Editor Tool - Read and edit files in the project
"""
from pathlib import Path
from typing import Dict, Any, Optional
from config import PROJECT_ROOT


class FileEditor:
    """Read and edit project files"""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read a file from the project

        Args:
            file_path: Relative path from project root

        Returns:
            Dict with 'success', 'content', 'error'
        """
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return {
                    "success": False,
                    "content": "",
                    "error": f"File not found: {file_path}",
                }

            # Security check - ensure path is within project
            if not str(full_path.resolve()).startswith(str(self.project_root.resolve())):
                return {
                    "success": False,
                    "content": "",
                    "error": "Access denied: Path outside project directory",
                }

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "success": True,
                "content": content,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Error reading file: {str(e)}",
            }

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file

        Args:
            file_path: Relative path from project root
            content: Content to write

        Returns:
            Dict with 'success', 'error'
        """
        try:
            full_path = self.project_root / file_path

            # Security check
            if not str(full_path.resolve()).startswith(str(self.project_root.resolve())):
                return {
                    "success": False,
                    "error": "Access denied: Path outside project directory",
                }

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error writing file: {str(e)}",
            }

    def get_read_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for reading files"""
        from utils.llm_client import create_tool_definition

        return create_tool_definition(
            name="read_file",
            description="Read the contents of a file in the project. Use this to inspect CMakeLists.txt, source files, or configuration files.",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file from project root (e.g., 'CMakeLists.txt', 'cyxwiz-engine/CMakeLists.txt')",
                },
            },
        )

    def get_write_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for writing files"""
        from utils.llm_client import create_tool_definition

        return create_tool_definition(
            name="write_file",
            description="Write content to a file in the project. Use this to fix configuration files, update CMakeLists.txt, or create missing files. BE CAREFUL - this overwrites the file!",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Relative path to the file from project root",
                },
                "content": {
                    "type": "string",
                    "description": "Complete content to write to the file",
                },
            },
        )

    def handle_read_tool_call(self, arguments: Dict[str, Any]) -> str:
        """Handle read_file tool call"""
        file_path = arguments.get("file_path", "")
        result = self.read_file(file_path)

        if result["success"]:
            return f"File contents of {file_path}:\n\n{result['content']}"
        else:
            return f"Error reading file: {result['error']}"

    def handle_write_tool_call(self, arguments: Dict[str, Any]) -> str:
        """Handle write_file tool call"""
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")
        result = self.write_file(file_path, content)

        if result["success"]:
            return f"Successfully wrote to {file_path}"
        else:
            return f"Error writing file: {result['error']}"


# Singleton instance
file_editor = FileEditor()
