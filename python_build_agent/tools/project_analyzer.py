"""
Project Analyzer Tool - Analyze project structure and find build instructions
"""
from pathlib import Path
from typing import Dict, Any, List
from config import PROJECT_ROOT


class ProjectAnalyzer:
    """Analyze project structure and find documentation"""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root

    def find_documentation(self) -> Dict[str, Any]:
        """
        Find README files and build documentation

        Returns:
            Dict with 'readme_files', 'build_files', 'docs_found'
        """
        readme_patterns = ["README.md", "README.txt", "README", "readme.md"]
        build_files = [
            "CMakeLists.txt",
            "Cargo.toml",
            "package.json",
            "pom.xml",
            "build.gradle",
            "Makefile",
            "meson.build",
        ]

        found_readmes = []
        found_build_files = []

        # Search for README files
        for pattern in readme_patterns:
            readme_path = self.project_root / pattern
            if readme_path.exists():
                found_readmes.append(str(pattern))

        # Search for build system files
        for build_file in build_files:
            build_path = self.project_root / build_file
            if build_path.exists():
                found_build_files.append(str(build_file))

        return {
            "readme_files": found_readmes,
            "build_files": found_build_files,
            "docs_found": len(found_readmes) > 0,
        }

    def list_directory(self, dir_path: str = ".") -> Dict[str, Any]:
        """
        List contents of a directory

        Args:
            dir_path: Relative path from project root

        Returns:
            Dict with 'files' and 'directories'
        """
        try:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                return {
                    "success": False,
                    "files": [],
                    "directories": [],
                    "error": f"Directory not found: {dir_path}",
                }

            files = []
            directories = []

            for item in full_path.iterdir():
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    directories.append(item.name)

            return {
                "success": True,
                "files": sorted(files),
                "directories": sorted(directories),
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "files": [],
                "directories": [],
                "error": f"Error listing directory: {str(e)}",
            }

    def get_find_docs_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for finding documentation"""
        from utils.llm_client import create_tool_definition

        return create_tool_definition(
            name="find_documentation",
            description="Find README files and build system files in the project. Use this first to understand the project structure and find build instructions.",
            parameters={},
        )

    def get_list_dir_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for listing directory"""
        from utils.llm_client import create_tool_definition

        return create_tool_definition(
            name="list_directory",
            description="List files and directories in a project directory. Use this to explore the project structure.",
            parameters={
                "dir_path": {
                    "type": "string",
                    "description": "Relative path to directory (default: '.' for root)",
                },
            },
        )

    def handle_find_docs_call(self, arguments: Dict[str, Any]) -> str:
        """Handle find_documentation tool call"""
        result = self.find_documentation()

        output = "Project Analysis:\n\n"

        if result["readme_files"]:
            output += f"README files found: {', '.join(result['readme_files'])}\n"
        else:
            output += "No README files found\n"

        if result["build_files"]:
            output += f"Build system files found: {', '.join(result['build_files'])}\n"
        else:
            output += "No build system files found\n"

        output += "\nRecommendation: Read the README.md file to understand how to build this project."

        return output

    def handle_list_dir_call(self, arguments: Dict[str, Any]) -> str:
        """Handle list_directory tool call"""
        dir_path = arguments.get("dir_path", ".")
        result = self.list_directory(dir_path)

        if result["success"]:
            output = f"Contents of {dir_path}:\n\n"
            if result["directories"]:
                output += f"Directories ({len(result['directories'])}):\n"
                output += "\n".join(f"  📁 {d}" for d in result["directories"])
                output += "\n\n"
            if result["files"]:
                output += f"Files ({len(result['files'])}):\n"
                output += "\n".join(f"  📄 {f}" for f in result["files"])
            return output
        else:
            return f"Error: {result['error']}"


# Singleton instance
project_analyzer = ProjectAnalyzer()
