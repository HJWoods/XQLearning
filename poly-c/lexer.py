"""
Lexer module for the Poly-c compiler
This module handles lexical analysis (tokenizing) for Poly-c code
"""

import re
from enum import Enum, auto

class TokenType(Enum):
    # Keywords
    INPUT = auto()
    ACTION = auto()
    CONST = auto()
    VAR = auto()
    ENV = auto()
    CONSTRAINTS = auto()
    GOALS = auto()
    IF = auto()
    ELSE = auto()
    RETURN = auto()
    MIN = auto()
    MAX = auto()
    WHILE = auto()
    
    # Types
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    CHAR = auto()
    
    # Symbols
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()
    HASH = auto()
    QUESTION = auto()     # ?
    COLON = auto()        # :
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MOD = auto()
    ASSIGN = auto()
    EQUALS = auto()
    NOT_EQUALS = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Boolean operators
    AND = auto()    # &&
    OR = auto()     # ||
    NOT = auto()    # !
    
    # Bitwise operators
    BIT_AND = auto() # &
    BIT_OR = auto()  # |
    BIT_XOR = auto() # ^
    BIT_NOT = auto() # ~
    BIT_LSHIFT = auto() # <<
    BIT_RSHIFT = auto() # >>
    
    # Other tokens
    IDENTIFIER = auto()
    NUMBER = auto()
    COMMENT = auto()
    EOF = auto()

class Token:
    """Represents a token from the source code"""
    
    def __init__(self, token_type, value, line, column):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
    
    def __str__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"
    
    def __repr__(self):
        return self.__str__()

class LexerError(Exception):
    """Exception raised for lexical errors during tokenization."""
    
    def __init__(self, message, line, column):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, column {column}: {message}")

class Lexer:
    """The lexical analyzer for Poly-c"""
    
    def __init__(self, source_code):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.errors = []
        
        # Define keywords mapping
        self.keywords = {
            'input': TokenType.INPUT,
            'action': TokenType.ACTION,
            'const': TokenType.CONST,
            'var': TokenType.VAR,
            'env': TokenType.ENV,
            'constraints': TokenType.CONSTRAINTS,
            'goals': TokenType.GOALS,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'return': TokenType.RETURN,
            'min': TokenType.MIN,
            'max': TokenType.MAX,
            'while': TokenType.WHILE,
            'float': TokenType.FLOAT,
            'int': TokenType.INT,
            'bool': TokenType.BOOL,
            'char': TokenType.CHAR
        }
    
    def current_char(self):
        """Get the current character."""
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def advance(self):
        """Move to the next character."""
        char = self.current_char()
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def peek(self, offset=1):
        """Look ahead without advancing."""
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def skip_whitespace(self):
        """Skip whitespace characters."""
        while self.current_char() and self.current_char().isspace():
            self.advance()
    
    def skip_comment(self):
        """Skip comments starting with #."""
        # Skip the # character
        self.advance()
        
        # Continue until end of line or end of file
        while self.current_char() is not None and self.current_char() != '\n':
            self.advance()
    
    def scan_identifier(self):
        """Process identifiers and keywords."""
        start_line = self.line
        start_column = self.column
        result = ''
        
        # Scan the identifier
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            result += self.advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(result, TokenType.IDENTIFIER)
        
        return Token(token_type, result, start_line, start_column)
    
    def scan_number(self):
        """Process numeric literals (integers and floats)."""
        start_line = self.line
        start_column = self.column
        result = ''
        is_float = False
        
        # Scan the number
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                if is_float:  # Second decimal point is an error
                    self.errors.append(LexerError("Invalid number format: multiple decimal points", 
                                                 self.line, self.column))
                is_float = True
            
            result += self.advance()
        
        # Convert to appropriate type
        try:
            value = float(result) if is_float else int(result)
            return Token(TokenType.NUMBER, value, start_line, start_column)
        except ValueError:
            self.errors.append(LexerError(f"Invalid number format: {result}", 
                                         start_line, start_column))
            return Token(TokenType.NUMBER, 0, start_line, start_column)
    
    def tokenize(self):
        """Tokenize the entire source code."""
        tokens = []
        
        while self.position < len(self.source):
            char = self.current_char()
            
            # Skip whitespace
            if char and char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if char == '#':
                self.skip_comment()
                continue
            
            # End of file
            if char is None:
                break
            
            start_line = self.line
            start_column = self.column
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                tokens.append(self.scan_identifier())
                continue
            
            # Numbers
            if char.isdigit():
                tokens.append(self.scan_number())
                continue
            
            # Two-character operators
            if char == '&' and self.peek() == '&':
                self.advance()  # Skip current &
                self.advance()  # Skip next &
                tokens.append(Token(TokenType.AND, '&&', start_line, start_column))
                continue
            
            if char == '|' and self.peek() == '|':
                self.advance()  # Skip current |
                self.advance()  # Skip next |
                tokens.append(Token(TokenType.OR, '||', start_line, start_column))
                continue
            
            if char == '=' and self.peek() == '=':
                self.advance()  # Skip current =
                self.advance()  # Skip next =
                tokens.append(Token(TokenType.EQUALS, '==', start_line, start_column))
                continue
            
            if char == '!' and self.peek() == '=':
                self.advance()  # Skip current !
                self.advance()  # Skip next =
                tokens.append(Token(TokenType.NOT_EQUALS, '!=', start_line, start_column))
                continue
            
            if char == '<' and self.peek() == '=':
                self.advance()  # Skip current <
                self.advance()  # Skip next =
                tokens.append(Token(TokenType.LESS_EQUAL, '<=', start_line, start_column))
                continue
            
            if char == '>' and self.peek() == '=':
                self.advance()  # Skip current >
                self.advance()  # Skip next =
                tokens.append(Token(TokenType.GREATER_EQUAL, '>=', start_line, start_column))
                continue
            
            if char == '<' and self.peek() == '<':
                self.advance()  # Skip current <
                self.advance()  # Skip next <
                tokens.append(Token(TokenType.BIT_LSHIFT, '<<', start_line, start_column))
                continue
            
            if char == '>' and self.peek() == '>':
                self.advance()  # Skip current >
                self.advance()  # Skip next >
                tokens.append(Token(TokenType.BIT_RSHIFT, '>>', start_line, start_column))
                continue
            
            # Single-character tokens
            token_type = None
            
            if char == '+':
                token_type = TokenType.PLUS
            elif char == '-':
                token_type = TokenType.MINUS
            elif char == '*':
                token_type = TokenType.MULTIPLY
            elif char == '/':
                token_type = TokenType.DIVIDE
            elif char == '%':
                token_type = TokenType.MOD
            elif char == '=':
                token_type = TokenType.ASSIGN
            elif char == '<':
                token_type = TokenType.LESS_THAN
            elif char == '>':
                token_type = TokenType.GREATER_THAN
            elif char == '!':
                token_type = TokenType.NOT
            elif char == '&':
                token_type = TokenType.BIT_AND
            elif char == '|':
                token_type = TokenType.BIT_OR
            elif char == '^':
                token_type = TokenType.BIT_XOR
            elif char == '~':
                token_type = TokenType.BIT_NOT
            elif char == '{':
                token_type = TokenType.LEFT_BRACE
            elif char == '}':
                token_type = TokenType.RIGHT_BRACE
            elif char == '[':
                token_type = TokenType.LEFT_BRACKET
            elif char == ']':
                token_type = TokenType.RIGHT_BRACKET
            elif char == '(':
                token_type = TokenType.LEFT_PAREN
            elif char == ')':
                token_type = TokenType.RIGHT_PAREN
            elif char == ',':
                token_type = TokenType.COMMA
            elif char == ';':
                token_type = TokenType.SEMICOLON
            elif char == '#':
                token_type = TokenType.HASH
            elif char == '?':
                token_type = TokenType.QUESTION
            elif char == ':':
                token_type = TokenType.COLON
            else:
                # Unrecognized character
                self.errors.append(LexerError(f"Unexpected character: '{char}'", 
                                             start_line, start_column))
                self.advance()
                continue
            
            tokens.append(Token(token_type, char, start_line, start_column))
            self.advance()
        
        # Add EOF token
        tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        
        return tokens, self.errors

def tokenize(source_code):
    """Tokenize poly-c source code."""
    lexer = Lexer(source_code)
    tokens, errors = lexer.tokenize()
    return tokens, errors