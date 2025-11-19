# Python Build Agent 🤖

An autonomous AI agent that can build **ANY** software project by reading its README and following the build instructions!

## 🌟 Features

- **Universal Project Support**: Works with CMake, Cargo, npm, Maven, Make, and any build system
- **README-Driven**: Reads project documentation to understand build process
- **Autonomous Error Resolution**: Automatically fixes common build errors
- **Multiple LLM Support**: Works with Claude (Anthropic) or GPT (OpenAI)
- **Interactive**: Shows step-by-step progress with beautiful terminal output
- **Safe**: Sandboxed execution within project directory

## 🚀 How It Works

1. **Discovery**: Finds README.md and build system files in the project
2. **Understanding**: Reads the README to learn how to build the project
3. **Execution**: Follows build instructions step by step
4. **Error Handling**: When errors occur, analyzes and fixes them automatically
5. **Iteration**: Repeats until build succeeds or max iterations reached

## 📋 Prerequisites

- Python 3.8+
- An API key from either:
  - [Anthropic](https://console.anthropic.com) (recommended)
  - [OpenAI](https://platform.openai.com)

## 🔧 Installation

1. **Install dependencies**:
   ```bash
   cd python_build_agent
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   ```bash
   # Copy example config
   cp .env.example .env

   # Edit .env and add your API key
   # For Claude:
   ANTHROPIC_API_KEY=your_key_here

   # For GPT:
   OPENAI_API_KEY=your_key_here
   ```

## 💻 Usage

### Build the Current Project

```bash
# Using Claude (default)
python main.py

# Using GPT
python main.py --provider openai
```

### Custom Goals

```bash
# Build with custom goal
python main.py --goal "build the release version"

# Setup and build
python main.py --goal "install dependencies and build the project"

# Run tests after building
python main.py --goal "build the project and run all tests"
```

### Advanced Options

```bash
python main.py --help

Options:
  --provider {anthropic,openai}
                        LLM provider to use (default: anthropic)
  --goal GOAL          Specific goal for the agent
  --max-iterations N   Maximum iterations to attempt (default: 10)
```

## 🛠️ Available Tools

The agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `find_documentation` | Finds README and build system files |
| `list_directory` | Explores project structure |
| `read_file` | Reads file contents |
| `write_file` | Creates or modifies files |
| `execute_command` | Runs shell commands |

## 📖 Example Output

```
╔════════════════════════════════════════════════╗
║     Python Build Agent - Autonomous Builder    ║
╚════════════════════════════════════════════════╝

🤖 Build Agent Starting

Goal: build the project successfully
Project: D:\Dev\CyxWiz_Claude

================================================================================
Iteration 1/10
================================================================================

🤖 Agent
┌──────────────────────────────────────────────────────────────┐
│ Let me start by finding the project documentation and build  │
│ system files to understand how to build this project.        │
└──────────────────────────────────────────────────────────────┘

🔧 Tool: find_documentation
Arguments: {}

📤 Tool Result: find_documentation
┌──────────────────────────────────────────────────────────────┐
│ Project Analysis:                                            │
│                                                              │
│ README files found: README.md                                │
│ Build system files found: CMakeLists.txt                     │
│                                                              │
│ Recommendation: Read the README.md file to understand how    │
│ to build this project.                                       │
└──────────────────────────────────────────────────────────────┘

...
```

## 🎯 Use Cases

### Building CyxWiz Project
```bash
python main.py
# Agent will:
# 1. Read README.md
# 2. See it's a CMake project
# 3. Run: cmake --preset windows-release
# 4. Run: cmake --build build/windows-release
# 5. Fix any errors encountered
```

### Building Any Other Project
```bash
cd /path/to/any/project
python /path/to/python_build_agent/main.py
# Agent will adapt to the project's build system!
```

### Examples:
- **Rust project**: Agent will find `Cargo.toml` and run `cargo build`
- **Node.js project**: Agent will find `package.json` and run `npm install && npm run build`
- **Python project**: Agent will find `setup.py` or `pyproject.toml` and run appropriate commands
- **Java project**: Agent will find `pom.xml` or `build.gradle` and use Maven/Gradle

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# LLM Settings
LLM_PROVIDER = "anthropic"  # or "openai"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_MODEL = "gpt-4o"

# Agent Behavior
MAX_ITERATIONS = 10  # Max attempts to fix errors
MAX_TOOL_CALLS = 50  # Max tool calls per iteration
TIMEOUT_SECONDS = 300  # Command timeout

# Project Settings
BUILD_PRESET = "windows-release"  # Default CMake preset
```

## 🔒 Security

- Agent only operates within the project directory
- File operations are sandboxed
- Commands run with limited permissions
- No access to system files outside project

## 🐛 Troubleshooting

### "API key not set" error
- Make sure you created `.env` file from `.env.example`
- Add your actual API key (not the placeholder text)

### Agent gets stuck in a loop
- Increase `--max-iterations`
- Check if the README has clear build instructions
- Manually fix obvious errors first

### Build still fails
- The agent will explain what it tried and why it failed
- You can review the conversation history
- Some complex errors may require manual intervention

## 🎨 Example Projects It Can Build

✅ CMake projects (C++, C)
✅ Rust projects (Cargo)
✅ Node.js projects (npm, yarn, pnpm)
✅ Python projects (pip, poetry, setuptools)
✅ Java projects (Maven, Gradle)
✅ Go projects
✅ Any project with clear README build instructions!

## 📝 How to Make Your Project Agent-Friendly

1. **Have a clear README.md** with build instructions
2. **List prerequisites** (dependencies, tools needed)
3. **Provide exact commands** to build
4. **Document common issues** and their solutions

## 🤝 Contributing

This is a fun meta-project! Feel free to:
- Add support for more build systems
- Improve error detection
- Add more tools for the agent
- Enhance the conversation flow

## 📄 License

Part of the CyxWiz project. Use freely for building your projects!

## 🌟 Fun Facts

- This agent can build itself! Just run it in the python_build_agent directory
- It learns from each error and adapts its strategy
- Uses the same AI that helps you code to automate builds
- Can work on any public GitHub repo with a README

---

Built with ❤️ by the CyxWiz team. Making CI/CD fun again! 🚀
