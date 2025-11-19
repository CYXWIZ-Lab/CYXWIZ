#!/bin/bash
# Quick start script for Python Build Agent

echo "╔════════════════════════════════════════════════╗"
echo "║     Python Build Agent - Quick Start          ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found! Please install Python 3.8+"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo ""
    echo "Please create .env file from .env.example:"
    echo "  1. Copy .env.example to .env"
    echo "  2. Add your ANTHROPIC_API_KEY or OPENAI_API_KEY"
    echo ""
    exit 1
fi

# Create virtual environment if needed
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting Build Agent..."
echo ""

python main.py "$@"
