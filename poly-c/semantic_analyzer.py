"""
Semantic Analyzer module for the Poly-c compiler
This module handles semantic analysis for Poly-c code
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
    
    def analyze(self):
        """Perform semantic analysis on the AST"""
        if not self.ast:
            return False, self.errors
        
        try:
            self._analyze_node(self.ast)
            
            # Check if main function exists
            if not self.has_main:
                self.errors.append("No 'main' function found. All poly-c programs must have a main function.")
            
            return len(self.errors) == 0, self.errors
        except Exception as e:
            if not isinstance(e, SemanticError):
                self.errors.append(f"Unexpected error during semantic analysis: {str(e)}")
            return False, self.errors
    
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
        # First pass to check for main function
        for child in node.children:
            if child.type == ASTNodeType.FUNCTION_DEF and child.value == "main":
                self.has_main = True
                break
        
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
            self.errors.append(SemanticError(f"Function '{function_name}' not found in symbol table",
                                         node.token.line if node.token else None))
            return
        
        # If this is the main function, verify it doesn't have parameters
        if function_name == "main" and function_symbol.parameters:
            self.errors.append(SemanticError("Main function should not have parameters",
                                         node.token.line if node.token else None))
        
        # Analyze the function body
        function_body = None
        for child in node.children:
            if child.type == ASTNodeType.BLOCK:
                function_body = child
                break
        
        if function_body:
            self._analyze_node(function_body)
        
        # If this is the 'main' function, verify it sets all actions
        if function_name == "main":
            self._verify_actions_set()
        
        self.current_function = None
    
    def _verify_actions_set(self):
        """Verify that all action variables are set in the main function"""
        action_symbols = self.symbol_table.get_symbols_by_type(SymbolType.ACTION)
        for action in action_symbols:
            if not any(ref[0] > 0 for ref in action.references):
                self.errors.append(SemanticError(
                    f"Action variable '{action.name}' is never set in main function",
                    action.defined_line
                ))
    
    def _analyze_constraints_block(self, node):
        """Analyze the constraints block"""
        for child in node.children:
            self._analyze_constraint(child)
    
    def _analyze_constraint(self, node):
        """Analyze a single constraint"""
        # Check that the constraint is a valid boolean expression
        for child in node.children:
            self._analyze_node(child)
            self._verify_no_env_variables(child)
    
    def _verify_no_env_variables(self, node):
        """Verify that no environment variables are used in constraints or policies"""
        if node.type == ASTNodeType.IDENTIFIER:
            var_name = node.value
            symbol = self.symbol_table.resolve(var_name)
            
            if symbol and symbol.symbol_type == SymbolType.ENV:
                self.errors.append(SemanticError(
                    f"Environment variable '{var_name}' cannot be used in constraints or policies",
                    node.token.line if node.token else None
                ))
        
        # Recursively check all children
        for child in node.children:
            self._verify_no_env_variables(child)
    
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
            
            # Condition should be a boolean expression
            if condition.type == ASTNodeType.LITERAL:
                if not isinstance(condition.value, bool):
                    self.errors.append(SemanticError(
                        "If condition should be a boolean expression",
                        condition.token.line if condition.token else None
                    ))
        
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
            
            # Condition should be a boolean expression
            if condition.type == ASTNodeType.LITERAL:
                if not isinstance(condition.value, bool):
                    self.errors.append(SemanticError(
                        "While condition should be a boolean expression",
                        condition.token.line if condition.token else None
                    ))
        
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
        
        # Check if main function is trying to return a value
        if self.current_function == "main" and node.children:
            self.errors.append(SemanticError(
                "Main function cannot return a value",
                node.token.line if node.token else None
            ))
        
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
        
        # Check if the function exists
        function_symbol = self.symbol_table.resolve(function_name)
        if not function_symbol:
            self.errors.append(SemanticError(
                f"Undefined function '{function_name}'",
                node.token.line if node.token else None
            ))
            return
        
        # Check if it's actually a function
        if function_symbol.symbol_type != SymbolType.FUNCTION:
            self.errors.append(SemanticError(
                f"'{function_name}' is not a function",
                node.token.line if node.token else None
            ))
            return
        
        # Check parameter count
        expected_params = len(function_symbol.parameters)
        actual_params = len(node.children)
        
        if expected_params != actual_params:
            self.errors.append(SemanticError(
                f"Function '{function_name}' expects {expected_params} parameters, got {actual_params}",
                node.token.line if node.token else None
            ))
        
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