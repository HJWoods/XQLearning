"""
Symbol Table module for the Poly-c compiler - FIXED VERSION
This module handles symbol management and scope tracking for Poly-c code
"""

from enum import Enum, auto

class SymbolType(Enum):
    INPUT = auto()
    ACTION = auto()
    CONST = auto()
    VAR = auto()
    ENV = auto()
    FUNCTION = auto()
    PARAMETER = auto()

class DataType(Enum):
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    CHAR = auto()
    VOID = auto()

class Symbol:
    """Represents a symbol in the symbol table"""
    
    def __init__(self, name, symbol_type, data_type=None, dimensions=None, parameters=None, line=None):
        self.name = name
        self.symbol_type = symbol_type  # INPUT, ACTION, CONST, etc.
        self.data_type = data_type      # INT, FLOAT, BOOL, etc.
        self.dimensions = dimensions or []  # For arrays
        self.parameters = parameters or []  # For functions
        self.defined_line = line
        self.references = []
    
    def __str__(self):
        type_str = self.data_type.name if self.data_type else 'unknown'
        if self.dimensions:
            for dim in self.dimensions:
                type_str += f"[{dim}]"
        
        extras = ""
        if self.symbol_type == SymbolType.FUNCTION:
            param_str = ", ".join(p.name for p in self.parameters)
            extras = f"({param_str})"
        
        return f"{self.symbol_type.name} {type_str} {self.name}{extras}"

class SymbolTableError(Exception):
    """Exception raised for symbol table errors."""
    
    def __init__(self, message, symbol=None):
        self.message = message
        self.symbol = symbol
        line_info = f" at line {symbol.defined_line}" if symbol and symbol.defined_line else ""
        super().__init__(f"{message}{line_info}")

class Scope:
    """Represents a scope in the symbol table"""
    
    def __init__(self, parent=None, name=None):
        self.symbols = {}
        self.parent = parent
        self.name = name or "unnamed"
    
    def define(self, symbol):
        """Define a symbol in this scope"""
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            
            # Allow redefinition if one is a function declaration and one is a full definition
            if (symbol.symbol_type == SymbolType.FUNCTION and 
                existing.symbol_type == SymbolType.FUNCTION):
                # Update the existing symbol with any new information
                if not existing.parameters and symbol.parameters:
                    existing.parameters = symbol.parameters
                return existing
            
            raise SymbolTableError(f"Symbol '{symbol.name}' redefined", existing)
        
        self.symbols[symbol.name] = symbol
        return symbol
    
    def resolve(self, name):
        """Resolve a symbol in this scope"""
        if name in self.symbols:
            return self.symbols[name]
        
        # Try parent scope if available
        if self.parent:
            return self.parent.resolve(name)
        
        return None

class SymbolTable:
    """Symbol table for managing scopes and symbols"""
    
    def __init__(self):
        self.global_scope = Scope(name="global")
        self.current_scope = self.global_scope
        self.scopes = [self.global_scope]
        self.all_symbols = {}  # For quick lookup of all symbols
        self.user_defined_functions = set()  # Track user-defined functions
    
    def enter_scope(self, name=None):
        """Enter a new scope"""
        new_scope = Scope(self.current_scope, name or f"scope_{len(self.scopes)}")
        self.scopes.append(new_scope)
        self.current_scope = new_scope
        return new_scope
    
    def exit_scope(self):
        """Exit the current scope"""
        if len(self.scopes) > 1:  # Ensure we don't exit global scope
            self.scopes.pop()
            self.current_scope = self.scopes[-1]
            return self.current_scope
        
        # Can't exit global scope
        return self.global_scope
    
    def define(self, symbol):
        """Define a symbol in the current scope"""
        try:
            self.current_scope.define(symbol)
            self.all_symbols[symbol.name] = symbol
            
            # Track function definitions
            if symbol.symbol_type == SymbolType.FUNCTION:
                self.user_defined_functions.add(symbol.name)
                
            return symbol
        except SymbolTableError as e:
            # Re-raise the exception
            raise
    
    def resolve(self, name):
        """Resolve a symbol by name"""
        return self.current_scope.resolve(name)
    
    def resolve_global(self, name):
        """Resolve a symbol in the global scope only"""
        return self.global_scope.symbols.get(name)
    
    def get_all_symbols(self):
        """Get all symbols defined in the table"""
        return list(self.all_symbols.values())
    
    def get_symbols_by_type(self, symbol_type):
        """Get all symbols of a specific type"""
        return [s for s in self.all_symbols.values() if s.symbol_type == symbol_type]
    
    def is_user_defined_function(self, name):
        """Check if a name is a user-defined function"""
        return name in self.user_defined_functions

def map_type_string(type_str):
    """Map a type string to a DataType enum"""
    type_map = {
        'int': DataType.INT,
        'float': DataType.FLOAT,
        'bool': DataType.BOOL,
        'char': DataType.CHAR
    }
    return type_map.get(type_str, DataType.FLOAT)  # Default to float if unknown

def extract_array_info(type_str):
    """Extract base type and dimensions from a type string"""
    import re
    
    # Extract base type (e.g., 'float' from 'float[3]')
    base_type = type_str.split('[')[0] if '[' in type_str else type_str
    
    # Extract dimensions (e.g., [3] from 'float[3]')
    dimensions = []
    dim_matches = re.findall(r'\[(\d+)\]', type_str)
    if dim_matches:
        dimensions = [int(dim) for dim in dim_matches]
    
    return base_type, dimensions

def build_symbol_table(ast):
    """Build a symbol table from the AST"""
    from parser import ASTNodeType
    
    symbol_table = SymbolTable()
    errors = []
    
    def process_node(node):
        """Process an AST node to build the symbol table"""
        if node is None:
            return
        
        try:
            if node.type == ASTNodeType.PROGRAM:
                # First, do a pre-scan to identify function definitions
                for child in node.children:
                    if child.type == ASTNodeType.FUNCTION_DEF:
                        # Register the function name so we know it exists
                        function_name = child.value
                        symbol = Symbol(
                            function_name, 
                            SymbolType.FUNCTION, 
                            DataType.FLOAT,  # Default return type
                            line=child.token.line if child.token else None
                        )
                        symbol_table.define(symbol)
                
                # Process all declarations
                for child in node.children:
                    if child.type == ASTNodeType.VARIABLE_DECL:
                        process_variable_declaration(child)
                
                # Process functions
                for child in node.children:
                    if child.type == ASTNodeType.FUNCTION_DEF:
                        process_function_definition(child)
                
                # Process blocks
                for child in node.children:
                    if child.type not in [ASTNodeType.VARIABLE_DECL, ASTNodeType.FUNCTION_DEF]:
                        process_node(child)
            
            elif node.type == ASTNodeType.CONSTRAINTS_BLOCK or node.type == ASTNodeType.GOALS_BLOCK:
                # Process all children
                for child in node.children:
                    process_node(child)
            
            elif node.type == ASTNodeType.BLOCK:
                # Create a new scope for the block
                symbol_table.enter_scope()
                
                # Process all statements in the block
                for child in node.children:
                    process_node(child)
                
                # Exit the block scope
                symbol_table.exit_scope()
            
            elif node.type == ASTNodeType.IF_STMT:
                # Process condition
                process_node(node.children[0])
                
                # Process then-block and else-block if present
                for i in range(1, len(node.children)):
                    process_node(node.children[i])
            
            elif node.type == ASTNodeType.WHILE_STMT:
                # Process condition and body
                for child in node.children:
                    process_node(child)
            
            elif node.type == ASTNodeType.RETURN_STMT:
                # Process return expression if present
                if node.children:
                    process_node(node.children[0])
            
            elif node.type == ASTNodeType.ASSIGN_STMT:
                # Process left and right sides
                for child in node.children:
                    process_node(child)
            
            elif node.type == ASTNodeType.BINARY_EXPR or node.type == ASTNodeType.UNARY_EXPR:
                # Process operands
                for child in node.children:
                    process_node(child)
            
            elif node.type == ASTNodeType.FUNCTION_CALL:
                # Check if the function exists
                function_name = node.value
                function_symbol = symbol_table.resolve(function_name)
                
                if not function_symbol:
                    # Only report an error if it's not a known function
                    if not symbol_table.is_user_defined_function(function_name):
                        errors.append(f"Line {node.token.line if node.token else '?'}: Undefined function '{function_name}'")
                elif function_symbol.symbol_type != SymbolType.FUNCTION:
                    errors.append(f"Line {node.token.line if node.token else '?'}: '{function_name}' is not a function")
                
                # Process arguments
                for child in node.children:
                    process_node(child)
            
            elif node.type == ASTNodeType.ARRAY_ACCESS:
                # Check if the array exists
                array_name = node.value
                array_symbol = symbol_table.resolve(array_name)
                if not array_symbol:
                    errors.append(f"Line {node.token.line if node.token else '?'}: Undefined array '{array_name}'")
                elif not array_symbol.dimensions:
                    errors.append(f"Line {node.token.line if node.token else '?'}: '{array_name}' is not an array")
                
                # Process index expression
                if node.children:
                    process_node(node.children[0])
            
            elif node.type == ASTNodeType.IDENTIFIER:
                # Check if the identifier exists
                symbol = symbol_table.resolve(node.value)
                if not symbol:
                    # Skip error for function calls, as they're handled separately
                    if not symbol_table.is_user_defined_function(node.value):
                        errors.append(f"Line {node.token.line if node.token else '?'}: Undefined variable '{node.value}'")
                else:
                    # Add a reference to the symbol
                    if node.token:
                        symbol.references.append((node.token.line, node.token.column))
        
        except Exception as e:
            errors.append(f"Symbol table error: {str(e)}")
    
    def process_variable_declaration(node):
        """Process a variable declaration node"""
        var_name = node.value
        var_type_node = node.children[0]
        data_type_node = node.children[1]
        
        # Map variable type
        var_type_map = {
            'input': SymbolType.INPUT,
            'action': SymbolType.ACTION,
            'const': SymbolType.CONST,
            'var': SymbolType.VAR,
            'env': SymbolType.ENV
        }
        symbol_type = var_type_map.get(var_type_node.value, SymbolType.VAR)
        
        # Parse data type and dimensions
        data_type_str = data_type_node.value
        base_type_str, dimensions = extract_array_info(data_type_str)
        data_type = map_type_string(base_type_str)
        
        # Create and add the symbol
        symbol = Symbol(var_name, symbol_type, data_type, dimensions, line=node.token.line)
        
        try:
            symbol_table.define(symbol)
        except SymbolTableError as e:
            errors.append(str(e))
        
        # Process initializer if present
        if len(node.children) > 2:
            process_node(node.children[2])
    
    def process_function_definition(node):
        """Process a function definition node"""
        function_name = node.value
        
        # Function symbol should already exist from the pre-scan
        function_symbol = symbol_table.resolve(function_name)
        
        # Enter a new scope for the function body
        symbol_table.enter_scope(function_name)
        
        # Process parameters
        parameters = []
        for child in node.children:
            if child.type == ASTNodeType.FUNCTION_PARAM:
                param_name = child.value
                param_symbol = Symbol(param_name, SymbolType.PARAMETER, DataType.FLOAT, line=child.token.line)
                
                try:
                    symbol_table.define(param_symbol)
                    parameters.append(param_symbol)
                except SymbolTableError as e:
                    errors.append(str(e))
        
        # Update function symbol with parameters if it exists
        if function_symbol:
            function_symbol.parameters = parameters
        
        # Process function body
        for child in node.children:
            if child.type == ASTNodeType.BLOCK:
                process_node(child)
        
        # Exit function scope
        symbol_table.exit_scope()
    
    # Start processing from the root
    if ast:
        process_node(ast)
    
    return symbol_table, errors

def build_from_ast(ast):
    """Build a symbol table from an AST"""
    return build_symbol_table(ast)