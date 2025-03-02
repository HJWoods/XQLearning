"""
Semantic Analyzer module for the Poly-c compiler - FIXED VERSION
This module handles semantic analysis for Poly-c code with enhanced function handling
"""

from symbol_table import SymbolType, DataType
from parser import ASTNodeType

class SemanticError(Exception):
    """Exception raised for semantic errors during analysis."""
    
    def __init__(self, message, line=None, column=None):
        self.message = message
        self.line = line
        self.column = column
        location = f" at line {line}" if line else ""
        super().__init__(f"{message}{location}")

class SemanticAnalyzer:
    """Semantic analyzer for Poly-c"""
    
    def __init__(self, ast, symbol_table):
        self.ast = ast
        self.symbol_table = symbol_table
        self.errors = []
        self.current_function = None
        self.has_main = False
        self.helper_functions = set([
            "relu", "sigmoid", "tanh", "leaky_relu", "identity", "abs", 
            "predict_intersection", "main"
        ])
        self.defined_functions = set()
    
    def analyze(self):
        """Perform semantic analysis on the AST"""
        if not self.ast:
            return False, self.errors
        
        try:
            # First pass: scan for all function definitions
            self._scan_functions(self.ast)
            
            # Check if main function exists
            if 'main' in self.defined_functions:
                self.has_main = True
            else:
                self.errors.append("No 'main' function found. All poly-c programs must have a main function.")
            
            # Second pass: full semantic analysis
            self._analyze_node(self.ast)
            
            return len(self.errors) == 0, self.errors
        except Exception as e:
            if not isinstance(e, SemanticError):
                self.errors.append(f"Unexpected error during semantic analysis: {str(e)}")
            return False, self.errors
    
    def _scan_functions(self, node):
        """First pass: scan for all function definitions"""
        if node is None:
            return
        
        if node.type == ASTNodeType.FUNCTION_DEF:
            function_name = node.value
            self.defined_functions.add(function_name)
            
            # Also register this function in the symbol table if not already there
            if not self.symbol_table.resolve(function_name):
                # For the case where a function is defined but not in symbol table
                from symbol_table import Symbol
                function_symbol = Symbol(function_name, SymbolType.FUNCTION, DataType.FLOAT, 
                                        line=node.token.line if node.token else None)
                self.symbol_table.define(function_symbol)
            
            # Make sure it's in the user-defined functions set too
            self.symbol_table.user_defined_functions.add(function_name)
        
        # Continue scanning all children
        for child in node.children:
            self._scan_functions(child)
    
    def _analyze_node(self, node):
        """Analyze a single AST node"""
        if node is None:
            return
        
        if node.type == ASTNodeType.PROGRAM:
            self._analyze_program(node)
        elif node.type == ASTNodeType.VARIABLE_DECL:
            self._analyze_variable_declaration(node)
        elif node.type == ASTNodeType.FUNCTION_DEF:
            self._analyze_function_definition(node)
        elif node.type == ASTNodeType.CONSTRAINTS_BLOCK:
            self._analyze_constraints_block(node)
        elif node.type == ASTNodeType.GOALS_BLOCK:
            self._analyze_goals_block(node)
        elif node.type == ASTNodeType.BLOCK:
            self._analyze_block(node)
        elif node.type == ASTNodeType.IF_STMT:
            self._analyze_if_statement(node)
        elif node.type == ASTNodeType.WHILE_STMT:
            self._analyze_while_statement(node)
        elif node.type == ASTNodeType.ASSIGN_STMT:
            self._analyze_assignment(node)
        elif node.type == ASTNodeType.RETURN_STMT:
            self._analyze_return_statement(node)
        elif node.type == ASTNodeType.BINARY_EXPR:
            self._analyze_binary_expression(node)
        elif node.type == ASTNodeType.UNARY_EXPR:
            self._analyze_unary_expression(node)
        elif node.type == ASTNodeType.FUNCTION_CALL:
            self._analyze_function_call(node)
        elif node.type == ASTNodeType.ARRAY_ACCESS:
            self._analyze_array_access(node)
        elif node.type == ASTNodeType.CONDITIONAL_EXPR:
            self._analyze_conditional_expression(node)
        else:
            # Recursively analyze all children
            for child in node.children:
                self._analyze_node(child)
    
    def _analyze_program(self, node):
        """Analyze the program node"""
        # Analyze all declarations and functions
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_variable_declaration(self, node):
        """Analyze a variable declaration"""
        # The variable should already be in the symbol table
        var_name = node.value
        symbol = self.symbol_table.resolve(var_name)
        
        if not symbol:
            self.errors.append(SemanticError(f"Symbol '{var_name}' not found in symbol table", 
                                         node.token.line if node.token else None))
            return
        
        # If there's an initializer, check that it's compatible with the variable type
        if len(node.children) > 2:
            initializer = node.children[2]
            self._analyze_node(initializer)
    
    def _analyze_function_definition(self, node):
        """Analyze a function definition"""
        function_name = node.value
        self.current_function = function_name
        
        # Function should already be in the symbol table
        function_symbol = self.symbol_table.resolve(function_name)
        if not function_symbol:
            # Register the function if it wasn't in the symbol table
            from symbol_table import Symbol
            function_symbol = Symbol(function_name, SymbolType.FUNCTION, DataType.FLOAT, 
                                    line=node.token.line if node.token else None)
            self.symbol_table.define(function_symbol)
            
            # Make sure it's in the user-defined functions set too
            self.symbol_table.user_defined_functions.add(function_name)
        
        # Analyze the function body
        function_body = None
        for child in node.children:
            if child.type == ASTNodeType.BLOCK:
                function_body = child
                break
        
        if function_body:
            self._analyze_node(function_body)
        
        self.current_function = None
    
    def _analyze_constraints_block(self, node):
        """Analyze the constraints block"""
        for child in node.children:
            self._analyze_constraint(child)
    
    def _analyze_constraint(self, node):
        """Analyze a single constraint"""
        # Check that the constraint is a valid boolean expression
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_goals_block(self, node):
        """Analyze the goals block"""
        for child in node.children:
            self._analyze_goal(child)
    
    def _analyze_goal(self, node):
        """Analyze a single goal"""
        # Goals can use environment variables, so no need to check for that
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_block(self, node):
        """Analyze a block of statements"""
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_if_statement(self, node):
        """Analyze an if statement"""
        # Analyze condition
        if node.children:
            condition = node.children[0]
            self._analyze_node(condition)
        
        # Analyze then-block
        if len(node.children) > 1:
            self._analyze_node(node.children[1])
        
        # Analyze else-block if present
        if len(node.children) > 2:
            self._analyze_node(node.children[2])
    
    def _analyze_while_statement(self, node):
        """Analyze a while statement"""
        # Analyze condition
        if node.children:
            condition = node.children[0]
            self._analyze_node(condition)
        
        # Analyze body
        if len(node.children) > 1:
            self._analyze_node(node.children[1])
    
    def _analyze_assignment(self, node):
        """Analyze an assignment statement"""
        if len(node.children) < 2:
            return
        
        left = node.children[0]
        right = node.children[1]
        
        # Analyze both sides
        self._analyze_node(left)
        self._analyze_node(right)
        
        # Check if the left side is something that can be assigned to
        if left.type == ASTNodeType.IDENTIFIER:
            var_name = left.value
            symbol = self.symbol_table.resolve(var_name)
            
            if not symbol:
                self.errors.append(SemanticError(
                    f"Undefined variable '{var_name}'",
                    left.token.line if left.token else None
                ))
                return
            
            # Check if the variable is readonly
            if symbol.symbol_type == SymbolType.CONST:
                self.errors.append(SemanticError(
                    f"Cannot assign to const variable '{var_name}'",
                    left.token.line if left.token else None
                ))
            
            # Cannot assign to input variables
            elif symbol.symbol_type == SymbolType.INPUT:
                self.errors.append(SemanticError(
                    f"Cannot assign to input variable '{var_name}'",
                    left.token.line if left.token else None
                ))
            
            # Cannot assign to environment variables
            elif symbol.symbol_type == SymbolType.ENV:
                self.errors.append(SemanticError(
                    f"Cannot assign to environment variable '{var_name}'",
                    left.token.line if left.token else None
                ))
        
        # Check array access
        elif left.type == ASTNodeType.ARRAY_ACCESS:
            array_name = left.value
            symbol = self.symbol_table.resolve(array_name)
            
            if not symbol:
                self.errors.append(SemanticError(
                    f"Undefined array '{array_name}'",
                    left.token.line if left.token else None
                ))
                return
            
            # Check if it's an array
            if not symbol.dimensions:
                self.errors.append(SemanticError(
                    f"Cannot index non-array variable '{array_name}'",
                    left.token.line if left.token else None
                ))
            
            # Check if the array is readonly
            if symbol.symbol_type == SymbolType.CONST:
                self.errors.append(SemanticError(
                    f"Cannot assign to const array '{array_name}'",
                    left.token.line if left.token else None
                ))
            
            # Cannot assign to input arrays
            elif symbol.symbol_type == SymbolType.INPUT:
                self.errors.append(SemanticError(
                    f"Cannot assign to input array '{array_name}'",
                    left.token.line if left.token else None
                ))
            
            # Cannot assign to environment arrays
            elif symbol.symbol_type == SymbolType.ENV:
                self.errors.append(SemanticError(
                    f"Cannot assign to environment array '{array_name}'",
                    left.token.line if left.token else None
                ))
        
        else:
            self.errors.append(SemanticError(
                "Invalid assignment target",
                left.token.line if left.token else None
            ))
    
    def _analyze_return_statement(self, node):
        """Analyze a return statement"""
        # Check if in a function context
        if not self.current_function:
            self.errors.append(SemanticError(
                "Return statement outside of function",
                node.token.line if node.token else None
            ))
            return
        
        # Analyze return expression if present
        if node.children:
            self._analyze_node(node.children[0])
    
    def _analyze_binary_expression(self, node):
        """Analyze a binary expression"""
        # Analyze operands
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_unary_expression(self, node):
        """Analyze a unary expression"""
        # Analyze operand
        if node.children:
            self._analyze_node(node.children[0])
    
    def _analyze_conditional_expression(self, node):
        """Analyze a conditional (ternary) expression"""
        # Analyze condition, then-expr, and else-expr
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_function_call(self, node):
        """Analyze a function call"""
        function_name = node.value
        
        # Check if the function exists or is a known helper function
        if not self.symbol_table.resolve(function_name) and \
           not function_name in self.defined_functions and \
           not function_name in self.helper_functions:
            self.errors.append(SemanticError(
                f"Undefined function '{function_name}'",
                node.token.line if node.token else None
            ))
            return
        
        # Analyze each argument
        for child in node.children:
            self._analyze_node(child)
    
    def _analyze_array_access(self, node):
        """Analyze an array access"""
        array_name = node.value
        
        # Check if the array exists
        array_symbol = self.symbol_table.resolve(array_name)
        if not array_symbol:
            self.errors.append(SemanticError(
                f"Undefined array '{array_name}'",
                node.token.line if node.token else None
            ))
            return
        
        # Check if it's an array
        if not array_symbol.dimensions:
            self.errors.append(SemanticError(
                f"Cannot index non-array variable '{array_name}'",
                node.token.line if node.token else None
            ))
        
        # Analyze index expression
        if node.children:
            self._analyze_node(node.children[0])

def analyze(ast, symbol_table):
    """Perform semantic analysis on an AST with a given symbol table"""
    analyzer = SemanticAnalyzer(ast, symbol_table)
    success, errors = analyzer.analyze()
    return success, errors