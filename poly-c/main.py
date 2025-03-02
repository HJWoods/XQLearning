#!/usr/bin/env python3
"""
Poly-c Compiler
-------------
A compiler for the poly-c language for reinforcement learning.
"""

import os
import sys
import argparse
from lexer import tokenize
from parser import parse
from symbol_table import build_from_ast
from semantic_analyzer import analyze
from code_generator import generate

def format_error_context(source_code, line_number, context=2):
    """Format error context with line numbers and highlighting."""
    if not source_code or not line_number or line_number <= 0:
        return ""
    
    lines = source_code.split('\n')
    if line_number > len(lines):
        return ""
    
    # Get context lines
    start = max(0, line_number - context - 1)
    end = min(len(lines), line_number + context)
    
    # Format with line numbers and highlighting
    result = []
    for i in range(start, end):
        line_num = i + 1
        prefix = ">" if line_num == line_number else " "
        result.append(f"{prefix} {line_num:4d} | {lines[i]}")
    
    return '\n'.join(result)

def compile_file(input_file, output_file=None, verbose=False):
    """Compile a poly-c file to Python."""
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.py'
    
    # Initialize error tracking
    has_errors = False
    error_count = 0
    
    try:
        # Read the source code
        with open(input_file, 'r') as f:
            source_code = f.read()
        
        if verbose:
            print(f"Compiling {input_file} to {output_file}...")
        
        # Stage 1: Lexical Analysis
        if verbose:
            print("\nStage 1: Lexical Analysis")
        
        tokens, lexer_errors = tokenize(source_code)
        
        if lexer_errors:
            has_errors = True
            error_count += len(lexer_errors)
            print("LEXICAL ERRORS:")
            for error in lexer_errors:
                print(f"  {error}")
                if hasattr(error, 'line') and error.line:
                    context = format_error_context(source_code, error.line)
                    if context:
                        print(f"Code context:\n{context}\n")
        
        if verbose:
            print(f"  Generated {len(tokens)} tokens")
        
        # Stage 2: Syntax Analysis
        if verbose:
            print("\nStage 2: Syntax Analysis")
        
        ast, parser_errors = parse(tokens)
        
        if parser_errors:
            has_errors = True
            error_count += len(parser_errors)
            print("SYNTAX ERRORS:")
            for error in parser_errors:
                print(f"  {error}")
                if hasattr(error, 'token') and error.token and hasattr(error.token, 'line'):
                    context = format_error_context(source_code, error.token.line)
                    if context:
                        print(f"Code context:\n{context}\n")
        
        if not ast:
            print("Compilation failed: Could not build syntax tree.")
            return False
        
        if verbose:
            print("  Generated Abstract Syntax Tree")
        
        # Stage 3: Symbol Table Construction
        if verbose:
            print("\nStage 3: Symbol Table Construction")
        
        symbol_table, symbol_errors = build_from_ast(ast)
        
        if symbol_errors:
            has_errors = True
            error_count += len(symbol_errors)
            print("SYMBOL TABLE ERRORS:")
            for error in symbol_errors:
                print(f"  {error}")
                # Try to extract line number
                import re
                line_match = re.search(r'at line (\d+)', str(error))
                if line_match:
                    line = int(line_match.group(1))
                    context = format_error_context(source_code, line)
                    if context:
                        print(f"Code context:\n{context}\n")
        
        if verbose:
            print(f"  Added {len(symbol_table.get_all_symbols())} symbols to the symbol table")
        
        # Stage 4: Semantic Analysis
        if verbose:
            print("\nStage 4: Semantic Analysis")
        
        semantic_success, semantic_errors = analyze(ast, symbol_table)
        
        if semantic_errors:
            has_errors = True
            error_count += len(semantic_errors)
            print("SEMANTIC ERRORS:")
            for error in semantic_errors:
                print(f"  {error}")
                # Try to extract line number
                import re
                line_match = re.search(r'at line (\d+)', str(error))
                if line_match:
                    line = int(line_match.group(1))
                    context = format_error_context(source_code, line)
                    if context:
                        print(f"Code context:\n{context}\n")
        
        if verbose:
            if semantic_success:
                print("  No semantic errors found")
            else:
                print(f"  Found {len(semantic_errors)} semantic errors")
        
        # Don't generate code if there are errors
        if has_errors:
            print(f"Compilation failed with {error_count} errors.")
            return False
        
        # Stage 5: Code Generation
        if verbose:
            print("\nStage 5: Code Generation")
        
        output_code = generate(ast, symbol_table)
        
        if not output_code:
            print("Compilation failed: Could not generate code.")
            return False
        
        # Write the output file
        with open(output_file, 'w') as f:
            f.write(output_code)
        
        if verbose:
            print(f"  Generated {len(output_code.split(os.linesep))} lines of Python code")
            print(f"\nCompilation successful. Output written to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Unexpected error during compilation: {str(e)}")
        import traceback
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