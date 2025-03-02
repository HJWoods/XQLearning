#!/usr/bin/env python3
"""
Enhanced Error Handling for Poly-c Compiler
-----------------------------------------
This module provides improved error reporting for the poly-c compiler.
"""

import os
import sys
from enum import Enum, auto
import traceback

class ErrorType(Enum):
    LEXICAL = auto()    # Error during lexical analysis (tokenization)
    SYNTAX = auto()     # Error during syntax analysis (parsing)
    SEMANTIC = auto()   # Error during semantic analysis
    GENERAL = auto()    # General compilation errors

class CompilationError(Exception):
    """Exception raised for errors during compilation."""
    
    def __init__(self, message, error_type, line=None, column=None, code_snippet=None, filename=None):
        self.message = message
        self.error_type = error_type
        self.line = line
        self.column = column
        self.code_snippet = code_snippet
        self.filename = filename
        super().__init__(self.message)
    
    def __str__(self):
        """Return a formatted error message."""
        result = []
        
        # Add error type and location
        prefix = f"{self.error_type.name} ERROR"
        if self.filename:
            prefix += f" in {self.filename}"
        if self.line is not None:
            prefix += f" (line {self.line}"
            if self.column is not None:
                prefix += f", column {self.column}"
            prefix += ")"
        
        result.append(prefix)
        result.append("-" * len(prefix))
        
        # Add the error message
        result.append(f"{self.message}")
        
        # Add code snippet if available
        if self.code_snippet:
            result.append("\nCode context:")
            result.append(self.code_snippet)
        
        return "\n".join(result)

def get_source_context(source_code, line, context_lines=2):
    """
    Extract a snippet of code around the error line.
    
    Args:
        source_code: Complete source code
        line: Line number of the error (1-based)
        context_lines: Number of lines to show before and after the error line
    
    Returns:
        A formatted string with line numbers and code
    """
    if not source_code:
        return None
    
    lines = source_code.split("\n")
    if line <= 0 or line > len(lines):
        return None
    
    # Calculate range of lines to show
    start = max(1, line - context_lines)
    end = min(len(lines), line + context_lines)
    
    # Create the formatted context
    result = []
    for i in range(start, end + 1):
        prefix = ">" if i == line else " "
        result.append(f"{prefix} {i:4d} | {lines[i-1]}")
    
    return "\n".join(result)

def report_error(e, source_code=None, filename=None):
    """
    Format and report a compilation error.
    
    Args:
        e: The exception to report
        source_code: The complete source code (optional)
        filename: The source filename (optional)
    """
    if isinstance(e, CompilationError):
        # Already formatted error
        if source_code and e.line and not e.code_snippet:
            # Add code snippet if not already included
            e.code_snippet = get_source_context(source_code, e.line)
        if filename and not e.filename:
            e.filename = filename
        print(str(e), file=sys.stderr)
    
    elif isinstance(e, SyntaxError):
        # Convert Python's SyntaxError to our CompilationError
        line = e.lineno if hasattr(e, 'lineno') else None
        column = e.offset if hasattr(e, 'offset') else None
        
        error = CompilationError(
            message=str(e),
            error_type=ErrorType.SYNTAX,
            line=line,
            column=column,
            filename=filename
        )
        
        if source_code and line:
            error.code_snippet = get_source_context(source_code, line)
        
        print(str(error), file=sys.stderr)
    
    else:
        # Other exceptions
        print(f"ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc()

def wrap_lexer(lexer_cls):
    """
    Wrap the lexer class to add error handling.
    
    Args:
        lexer_cls: The original Lexer class
    
    Returns:
        A wrapped Lexer class with enhanced error handling
    """
    original_tokenize = lexer_cls.tokenize
    
    def enhanced_tokenize(self):
        try:
            return original_tokenize(self)
        except Exception as e:
            if not isinstance(e, CompilationError):
                e = CompilationError(
                    message=str(e),
                    error_type=ErrorType.LEXICAL,
                    line=self.line,
                    column=self.column
                )
            raise e
    
    lexer_cls.tokenize = enhanced_tokenize
    return lexer_cls

def wrap_parser(parser_cls):
    """
    Wrap the parser class to add error handling.
    
    Args:
        parser_cls: The original Parser class
    
    Returns:
        A wrapped Parser class with enhanced error handling
    """
    original_parse = parser_cls.parse
    
    def enhanced_parse(self):
        try:
            return original_parse(self)
        except SyntaxError as e:
            # Get the current token for line information
            current_token = self.current_token()
            line = current_token.line if current_token else None
            column = current_token.column if current_token else None
            
            raise CompilationError(
                message=str(e),
                error_type=ErrorType.SYNTAX,
                line=line,
                column=column
            )
        except Exception as e:
            if not isinstance(e, CompilationError):
                # Get the current token for line information
                current_token = self.current_token()
                line = current_token.line if current_token else None
                column = current_token.column if current_token else None
                
                e = CompilationError(
                    message=str(e),
                    error_type=ErrorType.SYNTAX,
                    line=line,
                    column=column
                )
            raise e
    
    parser_cls.parse = enhanced_parse
    return parser_cls

def enhance_compiler(compiler_module):
    """
    Enhance the compiler module with better error handling.
    
    Args:
        compiler_module: The compiler module to enhance
    """
    original_compile_file = compiler_module.compile_file
    
    def enhanced_compile_file(input_file, output_file=None, verbose=False):
        try:
            # Read the source code
            with open(input_file, 'r') as f:
                source_code = f.read()
            
            # Try to compile with the original function
            return original_compile_file(input_file, output_file, verbose)
        
        except Exception as e:
            # Report the error with context
            report_error(e, source_code, input_file)
            return False
    
    compiler_module.compile_file = enhanced_compile_file
    return compiler_module

# Example usage:
if __name__ == "__main__":
    # This would be used to test the error handling
    try:
        raise CompilationError(
            message="Undefined variable 'foo'",
            error_type=ErrorType.SEMANTIC,
            line=10,
            column=5,
            code_snippet="   9 | x = 5\n> 10 | y = foo + 3\n  11 | z = y * 2",
            filename="example.polyc"
        )
    except CompilationError as e:
        print(str(e))