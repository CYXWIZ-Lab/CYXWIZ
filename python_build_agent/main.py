#!/usr/bin/env python3
"""
Python Build Agent - Autonomous agent that builds projects

This agent can build ANY project by:
1. Reading the README.md to understand build instructions
2. Following those instructions step by step
3. Automatically fixing errors it encounters

Usage:
    python main.py [--provider anthropic|openai] [--goal "custom goal"]
"""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from agents.build_agent import BuildAgent

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Python Build Agent - Autonomous project builder"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Specific goal for the agent (default: build the project)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations to attempt (default: 10)",
    )

    args = parser.parse_args()

    # Update config with CLI args
    import config
    config.LLM_PROVIDER = args.provider
    config.MAX_ITERATIONS = args.max_iterations

    console.print("\n[bold blue]╔════════════════════════════════════════════════╗[/bold blue]")
    console.print("[bold blue]║     Python Build Agent - Autonomous Builder    ║[/bold blue]")
    console.print("[bold blue]╚════════════════════════════════════════════════╝[/bold blue]\n")

    console.print(f"Provider: [cyan]{args.provider}[/cyan]")
    console.print(f"Max Iterations: [cyan]{args.max_iterations}[/cyan]")
    console.print(f"Project: [cyan]{config.PROJECT_ROOT}[/cyan]\n")

    # Check API keys
    if args.provider == "anthropic" and not config.ANTHROPIC_API_KEY:
        console.print("[bold red]Error: ANTHROPIC_API_KEY not set![/bold red]")
        console.print("Set it in .env file or environment variable")
        sys.exit(1)
    elif args.provider == "openai" and not config.OPENAI_API_KEY:
        console.print("[bold red]Error: OPENAI_API_KEY not set![/bold red]")
        console.print("Set it in .env file or environment variable")
        sys.exit(1)

    # Create and run the agent
    agent = BuildAgent(llm_provider=args.provider)

    try:
        success = agent.build_project(goal=args.goal)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Build agent interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
