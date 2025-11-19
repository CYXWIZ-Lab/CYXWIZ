"""
Build Agent - Autonomous agent that builds projects by reading documentation
"""
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, PROJECT_ROOT
from utils.llm_client import LLMClient
from tools.command_executor import command_executor
from tools.file_editor import file_editor
from tools.project_analyzer import project_analyzer


console = Console()


class BuildAgent:
    """Autonomous agent that can build any project"""

    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMClient(provider=llm_provider)
        self.conversation_history: List[Dict[str, Any]] = []
        self.iteration = 0

        # Register tools
        self.tools = [
            project_analyzer.get_find_docs_tool_definition(),
            project_analyzer.get_list_dir_tool_definition(),
            file_editor.get_read_tool_definition(),
            file_editor.get_write_tool_definition(),
            command_executor.get_tool_definition(),
        ]

        self.tool_handlers = {
            "find_documentation": project_analyzer.handle_find_docs_call,
            "list_directory": project_analyzer.handle_list_dir_call,
            "read_file": file_editor.handle_read_tool_call,
            "write_file": file_editor.handle_write_tool_call,
            "execute_command": command_executor.handle_tool_call,
        }

    def build_project(self, goal: Optional[str] = None) -> bool:
        """
        Main method to build the project

        Args:
            goal: Optional specific goal (default: "build the project")

        Returns:
            True if build succeeded, False otherwise
        """
        if goal is None:
            goal = "build the project successfully"

        console.print(Panel.fit(
            f"🤖 Build Agent Starting\n\n"
            f"Goal: {goal}\n"
            f"Project: {PROJECT_ROOT}",
            title="Python Build Agent",
            border_style="blue",
        ))

        # Initial system message
        system_message = self._create_system_message(goal)
        self.conversation_history = [
            {"role": "system", "content": system_message}
        ]

        # Start the agent loop
        for iteration in range(MAX_ITERATIONS):
            self.iteration = iteration + 1

            console.print(f"\n{'='*80}")
            console.print(f"[bold blue]Iteration {self.iteration}/{MAX_ITERATIONS}[/bold blue]")
            console.print(f"{'='*80}\n")

            # Get agent's next action
            success = self._run_iteration()

            if success:
                console.print(Panel.fit(
                    "✅ Build completed successfully!",
                    title="Success",
                    border_style="green",
                ))
                return True

        console.print(Panel.fit(
            f"❌ Failed to complete build after {MAX_ITERATIONS} iterations",
            title="Failed",
            border_style="red",
        ))
        return False

    def _create_system_message(self, goal: str) -> str:
        """Create the system message for the agent"""
        return f"""You are an autonomous build agent. Your goal is to: {goal}

You are working on a project at: {PROJECT_ROOT}

IMPORTANT INSTRUCTIONS:
1. First, use find_documentation to discover README files and build system files
2. Read the README.md file to understand how to build the project
3. Follow the build instructions exactly as described in the README
4. If you encounter errors, analyze them carefully and fix them
5. Use the available tools to execute commands, read files, and make changes
6. Work step by step - don't try to do everything at once
7. After each command, check if it succeeded before proceeding

AVAILABLE TOOLS:
- find_documentation: Find README and build files
- list_directory: Explore project structure
- read_file: Read file contents
- write_file: Create or modify files (use carefully!)
- execute_command: Run shell commands

WORKFLOW:
1. Find and read documentation (README.md)
2. Identify the build system (CMake, Cargo, npm, etc.)
3. Install dependencies if needed
4. Configure the build
5. Execute the build
6. Handle any errors that occur

When you successfully build the project, respond with exactly: "BUILD_SUCCESS"
If you encounter an error you cannot fix, explain the issue clearly.

Start by finding the project documentation!"""

    def _run_iteration(self) -> bool:
        """Run one iteration of the agent loop"""
        try:
            # Get response from LLM
            response = self.llm.chat(
                messages=self.conversation_history,
                tools=self.tools,
                max_tokens=4096,
            )

            # Display agent's thinking
            if response["content"]:
                console.print(Panel(
                    Markdown(response["content"]),
                    title="🤖 Agent",
                    border_style="cyan",
                ))

            # Check for success
            if "BUILD_SUCCESS" in response["content"]:
                return True

            # Handle tool calls
            if response["tool_calls"]:
                tool_results = []

                for tool_call in response["tool_calls"]:
                    result = self._execute_tool(tool_call)
                    tool_results.append(result)

                # For Anthropic: Build content blocks with text + tool uses
                assistant_content = []
                if response["content"]:
                    assistant_content.append({
                        "type": "text",
                        "text": response["content"]
                    })
                for tool_call in response["tool_calls"]:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["arguments"],
                    })

                # Add assistant message
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # Add tool results as user message
                tool_result_content = []
                for tool_call, result in zip(response["tool_calls"], tool_results):
                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": result,
                    })

                self.conversation_history.append({
                    "role": "user",
                    "content": tool_result_content,
                })

            else:
                # No tool calls, add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response["content"],
                })

                # If no tool calls and no success, prompt agent to continue
                self.conversation_history.append({
                    "role": "user",
                    "content": "Please continue. What's your next step?",
                })

            return False

        except Exception as e:
            console.print(f"[bold red]Error in iteration:[/bold red] {str(e)}")
            return False

    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call"""
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        console.print(f"\n[bold yellow]🔧 Tool: {tool_name}[/bold yellow]")
        console.print(f"Arguments: {arguments}")

        # Get tool handler
        handler = self.tool_handlers.get(tool_name)
        if not handler:
            return f"Error: Unknown tool '{tool_name}'"

        # Execute tool
        try:
            result = handler(arguments)
            console.print(Panel(
                result[:500] + ("..." if len(result) > 500 else ""),
                title=f"📤 Tool Result: {tool_name}",
                border_style="green",
            ))
            return result

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            return error_msg
