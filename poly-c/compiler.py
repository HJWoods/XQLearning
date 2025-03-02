#!/usr/bin/env python3
"""
Poly-C Compiler with Improved Error Handling
------------------------------------------
A compiler for the poly-c reinforcement learning language.
Poly-c allows users to write reinforcement learning rewards, constraints, 
and policies all in one language.
"""

import os
import sys
import argparse
from lexer import tokenize_code
from parser import parse_tokens
from symbol_table import build_symbol_table
from semantic_analyzer import analyze_ast
from code_generator import generate_code

# Function to get a snippet of code around a specific line
def get_code_snippet(source_code, line, context=2):
    """Get a snippet of code around the specified line."""
    lines = source_code.split('\n')
    start = max(0, line - context - 1)
    end = min(len(lines), line + context)
    
    snippet = []
    for i in range(start, end):
        prefix = ">" if i == line - 1 else " "
        line_num = i + 1
        snippet.append(f"{prefix} {line_num:4d} | {lines[i]}")
    
    return '\n'.join(snippet)

def compile_file(input_file, output_file=None, verbose=False):
    """Compile a poly-c source file to Python code."""
    
    # Determine output file name if not specified
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.py"
    
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            source_code = f.read()
        
        if verbose:
            print(f"Compiling {input_file} to {output_file}...")
        
        # Stage 1: Lexical Analysis
        if verbose:
            print("Stage 1: Lexical Analysis")
        tokens = tokenize_code(source_code)
        if verbose:
            print(f"  Generated {len(tokens)} tokens")
        
        # Stage 2: Syntax Analysis (Parsing)
        if verbose:
            print("Stage 2: Syntax Analysis")
        ast = parse_tokens(tokens)
        if verbose:
            print("  Generated Abstract Syntax Tree")
        
        # Stage 3: Build Symbol Table 
        if verbose:
            print("Stage 3: Symbol Table Construction")
        # Build symbol table from AST instead of tokens
        symbol_table = build_symbol_table(ast)
        if verbose:
            symbols = symbol_table.get_all_symbols()
            print(f"  Added {len(symbols)} symbols to the symbol table")
        
        # Stage 4: Semantic Analysis
        if verbose:
            print("Stage 4: Semantic Analysis")
        success, errors, symbol_table = analyze_ast(ast, symbol_table)
        if not success:
            print(f"SEMANTIC ERROR in {input_file}")
            print("-" * 40)
            for error in errors:
                # Extract line number from error message
                import re
                line_match = re.search(r'Line (\d+):', error)
                if line_match:
                    line = int(line_match.group(1))
                    snippet = get_code_snippet(source_code, line)
                    print(f"{error}\n{snippet}")
                else:
                    print(f"{error}")
            return False
        if verbose:
            print("  No semantic errors found")
        
        # Stage 5: Code Generation
        if verbose:
            print("Stage 5: Code Generation")
        output_code = generate_code(ast, symbol_table)
        if verbose:
            print(f"  Generated {len(output_code.split(os.linesep))} lines of Python code")
        
        # Write the output file
        with open(output_file, 'w') as f:
            f.write(output_code)
        
        if verbose:
            print(f"Compilation successful. Output written to {output_file}")
        
        return True
    
    except Exception as e:
        import traceback
        
        # Get line information if available
        line = None
        if hasattr(e, 'line'):
            line = e.line
        elif hasattr(e, 'token') and hasattr(e.token, 'line'):
            line = e.token.line
        
        # Format the error message
        error_type = type(e).__name__
        error_message = str(e)
        
        print(f"{error_type} ERROR in {input_file}" + (f" (line {line})" if line else ""))
        print("-" * 40)
        print(f"{error_message}")
        
        # Add code snippet if we have a line number
        if line:
            snippet = get_code_snippet(source_code, line)
            print(f"\nCode context:\n{snippet}")
        
        if verbose:
            traceback.print_exc()
        
        return False

def main():
    """Main entry point for the compiler."""
    parser = argparse.ArgumentParser(description='Poly-c compiler')
    parser.add_argument('input', help='Input .polyc file')
    parser.add_argument('-o', '--output', help='Output Python file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    success = compile_file(args.input, args.output, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()