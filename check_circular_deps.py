#!/usr/bin/env python3
"""
Check for circular dependencies in header files
"""
import re
import os
from pathlib import Path
from collections import defaultdict

def extract_includes(file_path):
    """Extract all local includes from a header file"""
    includes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find all includes of the form #include "something.h"
            pattern = r'#include\s+"([^"]+)"'
            matches = re.findall(pattern, content)
            for match in matches:
                # Extract just the filename from the path
                includes.append(os.path.basename(match))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return includes

def build_dependency_graph(header_dir):
    """Build a dependency graph from header files"""
    graph = defaultdict(list)
    headers = list(Path(header_dir).glob("*.h"))

    for header_path in headers:
        header_name = header_path.name
        includes = extract_includes(header_path)
        graph[header_name] = includes

    return graph

def find_cycles(graph, start, path=None, visited=None):
    """DFS to find cycles in dependency graph"""
    if path is None:
        path = []
    if visited is None:
        visited = set()

    path = path + [start]
    visited.add(start)

    cycles = []
    if start in graph:
        for neighbor in graph[start]:
            if neighbor in path:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif neighbor not in visited:
                cycles.extend(find_cycles(graph, neighbor, path, visited))

    return cycles

def main():
    header_dir = Path(__file__).parent / "cyxwiz-backend" / "include" / "cyxwiz"

    print(f"Analyzing headers in: {header_dir}")
    print("=" * 70)

    graph = build_dependency_graph(header_dir)

    print("\nDependency Graph:")
    print("-" * 70)
    for header, includes in sorted(graph.items()):
        if includes:
            print(f"{header}:")
            for inc in includes:
                print(f"  -> {inc}")
        else:
            print(f"{header}: (no dependencies)")

    print("\n" + "=" * 70)
    print("Checking for circular dependencies...")
    print("-" * 70)

    all_cycles = []
    for header in graph.keys():
        cycles = find_cycles(graph, header)
        all_cycles.extend(cycles)

    # Remove duplicate cycles
    unique_cycles = []
    seen = set()
    for cycle in all_cycles:
        cycle_str = " -> ".join(cycle)
        if cycle_str not in seen:
            unique_cycles.append(cycle)
            seen.add(cycle_str)

    if unique_cycles:
        print(f"\nFOUND {len(unique_cycles)} CIRCULAR DEPENDENCIES:")
        for i, cycle in enumerate(unique_cycles, 1):
            print(f"\n{i}. {' -> '.join(cycle)}")
        return 1
    else:
        print("\nNO CIRCULAR DEPENDENCIES FOUND!")
        print("All headers are properly structured.")
        return 0

if __name__ == "__main__":
    exit(main())
