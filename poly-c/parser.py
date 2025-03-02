"""
Parser module for the Poly-c compiler
"""

from enum import Enum, auto
from lexer import TokenType

class ASTNodeType(Enum):
    PROGRAM = auto()
    VARIABLE_DECL = auto()
    CONSTRAINTS_BLOCK = auto()
    GOALS_BLOCK = auto()
    CONSTRAINT = auto()
    GOAL = auto()
    FUNCTION_DEF = auto()
    FUNCTION_PARAM = auto()
    BLOCK = auto()
    IF_STMT = auto()
    WHILE_STMT = auto()
    ASSIGN_STMT = auto()
    RETURN_STMT = auto()
    BINARY_EXPR = auto()
    UNARY_EXPR = auto()
    IDENTIFIER = auto()
    LITERAL = auto()
    FUNCTION_CALL = auto()
    ARRAY_ACCESS = auto()
    CONDITIONAL_EXPR = auto()  # Ternary expressions: condition ? expr1 : expr2

class ASTNode:
    """Represents a node in the Abstract Syntax Tree"""
    
    def __init__(self, node_type, value=None, token=None):
        self.type = node_type
        self.value = value
        self.token = token
        self.children = []
    
    def add_child(self, child):
        """Add a child node"""
        self.children.append(child)
        return child
    
    def __str__(self):
        value_str = f" '{self.value}'" if self.value is not None else ""
        return f"{self.type.name}{value_str}"

class ParserError(Exception):
    """Exception raised for syntax errors during parsing."""
    
    def __init__(self, message, token=None):
        self.message = message
        self.token = token
        line_info = f" at line {token.line}, column {token.column}" if token else ""
        super().__init__(f"{message}{line_info}")

class Parser:
    """The syntax analyzer for Poly-c"""
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
        self.errors = []
    
    def current_token(self):
        """Get the current token."""
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return self.tokens[-1]  # Return EOF token
    
    def advance(self):
        """Move to the next token."""
        token = self.current_token()
        if token.type != TokenType.EOF:
            self.current += 1
        return token
    
    def peek(self, offset=1):
        """Look ahead without advancing."""
        peek_index = self.current + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index]
        return self.tokens[-1]  # Return EOF token
    
    def match(self, *token_types):
        """Check if current token matches any of the expected types."""
        current = self.current_token()
        if current.type in token_types:
            return self.advance()
        return None
    
    def expect(self, token_type, error_message=None):
        """Expect a token of the given type, raise error if not found."""
        current = self.current_token()
        if current.type == token_type:
            return self.advance()
        
        if not error_message:
            error_message = f"Expected {token_type.name}, got {current.type.name}"
        
        error = ParserError(error_message, current)
        self.errors.append(error)
        raise error
    
    def parse(self):
        """Parse the entire program."""
        try:
            program_node = ASTNode(ASTNodeType.PROGRAM)
            
            # Process variable declarations, constraints, goals, and functions
            while self.current_token().type != TokenType.EOF:
                # Handle variable declarations
                if self.current_token().type in [
                    TokenType.INPUT, TokenType.ACTION, TokenType.CONST, 
                    TokenType.VAR, TokenType.ENV
                ]:
                    try:
                        var_decl = self.parse_variable_declaration()
                        program_node.add_child(var_decl)
                    except ParserError as e:
                        # Try to synchronize to the next declaration
                        self.synchronize_to_declaration()
                
                # Handle constraints block
                elif self.current_token().type == TokenType.CONSTRAINTS:
                    try:
                        self.advance()  # Consume 'constraints' token
                        constraints = self.parse_constraints_block()
                        program_node.add_child(constraints)
                    except ParserError as e:
                        # Try to synchronize to the next section
                        self.synchronize_to_section()
                
                # Handle goals block
                elif self.current_token().type == TokenType.GOALS:
                    try:
                        self.advance()  # Consume 'goals' token
                        goals = self.parse_goals_block()
                        program_node.add_child(goals)
                    except ParserError as e:
                        # Try to synchronize to the next section
                        self.synchronize_to_section()
                
                # Handle function definitions
                elif self.current_token().type == TokenType.IDENTIFIER:
                    try:
                        function = self.parse_function_definition()
                        program_node.add_child(function)
                    except ParserError as e:
                        # Try to synchronize to the next function
                        self.synchronize_to_function()
                
                else:
                    # Skip unknown token and report error
                    error = ParserError(f"Unexpected token: {self.current_token().type.name}", self.current_token())
                    self.errors.append(error)
                    self.advance()  # Skip the problematic token
            
            return program_node
        
        except Exception as e:
            if not isinstance(e, ParserError):
                error = ParserError(str(e), self.current_token())
                self.errors.append(error)
            return None
    
    def synchronize_to_declaration(self):
        """Recover from an error by advancing to the next variable declaration."""
        while (self.current_token().type != TokenType.EOF and 
               self.current_token().type not in [
                   TokenType.INPUT, TokenType.ACTION, TokenType.CONST, 
                   TokenType.VAR, TokenType.ENV, TokenType.CONSTRAINTS, 
                   TokenType.GOALS, TokenType.IDENTIFIER
               ]):
            self.advance()
    
    def synchronize_to_section(self):
        """Recover from an error by advancing to the next section."""
        while (self.current_token().type != TokenType.EOF and 
               self.current_token().type not in [
                   TokenType.CONSTRAINTS, TokenType.GOALS, TokenType.IDENTIFIER
               ]):
            self.advance()
    
    def synchronize_to_function(self):
        """Recover from an error by advancing to the next function."""
        nesting_level = 0
        while self.current_token().type != TokenType.EOF:
            if self.current_token().type == TokenType.LEFT_BRACE:
                nesting_level += 1
            elif self.current_token().type == TokenType.RIGHT_BRACE:
                nesting_level -= 1
                if nesting_level <= 0:
                    self.advance()  # Skip the closing brace
                    break
            self.advance()
    
    def parse_variable_declaration(self):
        """Parse a variable declaration."""
        # Parse variable type (input, action, const, var, env)
        var_type_token = self.advance()
        
        # Parse data type (float, int, bool, char)
        data_type_token = None
        array_dims = []
        
        # Handle the special case of env declarations that might omit type
        if var_type_token.type == TokenType.ENV:
            if self.current_token().type in [TokenType.FLOAT, TokenType.INT, TokenType.BOOL, TokenType.CHAR]:
                data_type_token = self.advance()
            else:
                # Default to float for env variables without type
                from lexer import Token
                data_type_token = Token(TokenType.FLOAT, "float", var_type_token.line, var_type_token.column)
        else:
            # Expect a data type for other variable categories
            try:
                data_type_token = self.expect(
                    TokenType.FLOAT, f"Expected data type after '{var_type_token.value}'"
                )
            except ParserError:
                # Try the other data types
                if self.current_token().type in [TokenType.INT, TokenType.BOOL, TokenType.CHAR]:
                    data_type_token = self.advance()
                else:
                    # Use a dummy token as we've already reported the error
                    from lexer import Token
                    data_type_token = Token(TokenType.FLOAT, "float", var_type_token.line, var_type_token.column)
        
        # Check for array dimensions
        if self.match(TokenType.LEFT_BRACKET):
            # Parse the dimension
            dim_token = self.expect(TokenType.NUMBER, "Expected number for array dimension")
            array_dims.append(int(dim_token.value))
            
            # Expect closing bracket
            self.expect(TokenType.RIGHT_BRACKET, "Expected ']' after array dimension")
        
        # Parse variable name
        name_token = self.expect(TokenType.IDENTIFIER, "Expected variable name")
        
        # Create the variable declaration node
        var_node = ASTNode(ASTNodeType.VARIABLE_DECL, name_token.value, var_type_token)
        
        # Add type information as children
        type_node = ASTNode(ASTNodeType.IDENTIFIER, var_type_token.value, var_type_token)
        var_node.add_child(type_node)
        
        data_type_value = data_type_token.value
        if array_dims:
            data_type_value = f"{data_type_value}[{array_dims[0]}]"
        
        data_type_node = ASTNode(ASTNodeType.IDENTIFIER, data_type_value, data_type_token)
        var_node.add_child(data_type_node)
        
        # Check for initialization
        if self.match(TokenType.ASSIGN):
            init_expr = self.parse_expression()
            var_node.add_child(init_expr)
        
        return var_node
    
    def parse_constraints_block(self):
        """Parse a constraints block."""
        constraints_node = ASTNode(ASTNodeType.CONSTRAINTS_BLOCK)
        
        # Expect opening bracket
        self.expect(TokenType.LEFT_BRACKET, "Expected '[' after 'constraints'")
        
        # Parse constraints until closing bracket
        while self.current_token().type != TokenType.RIGHT_BRACKET:
            if self.current_token().type == TokenType.EOF:
                raise ParserError("Unexpected end of file in constraints block", self.current_token())
            
            constraint = self.parse_constraint()
            constraints_node.add_child(constraint)
        
        # Expect closing bracket
        self.expect(TokenType.RIGHT_BRACKET, "Expected ']' to close constraints block")
        
        return constraints_node
    
    def parse_constraint(self):
        """Parse a single constraint."""
        constraint_node = ASTNode(ASTNodeType.CONSTRAINT)
        
        # Parse the constraint expression
        expr = self.parse_expression()
        constraint_node.add_child(expr)
        
        return constraint_node
    
    def parse_goals_block(self):
        """Parse a goals block."""
        goals_node = ASTNode(ASTNodeType.GOALS_BLOCK)
        
        # Expect opening bracket
        self.expect(TokenType.LEFT_BRACKET, "Expected '[' after 'goals'")
        
        # Parse goals until closing bracket
        while self.current_token().type != TokenType.RIGHT_BRACKET:
            if self.current_token().type == TokenType.EOF:
                raise ParserError("Unexpected end of file in goals block", self.current_token())
            
            goal = self.parse_goal()
            goals_node.add_child(goal)
        
        # Expect closing bracket
        self.expect(TokenType.RIGHT_BRACKET, "Expected ']' to close goals block")
        
        return goals_node
    
    def parse_goal(self):
        """Parse a single goal."""
        goal_node = ASTNode(ASTNodeType.GOAL)
        
        # Check for min/max goal
        if self.match(TokenType.MIN, TokenType.MAX):
            min_max_token = self.tokens[self.current - 1]
            goal_node.value = min_max_token.value
            goal_node.token = min_max_token
        
        # Parse the goal expression
        expr = self.parse_expression()
        goal_node.add_child(expr)
        
        return goal_node
    
    def parse_function_definition(self):
        """Parse a function definition."""
        # Parse function name
        function_name_token = self.expect(TokenType.IDENTIFIER, "Expected function name")
        
        # Create function node
        function_node = ASTNode(ASTNodeType.FUNCTION_DEF, function_name_token.value, function_name_token)
        
        # Parse parameters
        self.expect(TokenType.LEFT_PAREN, f"Expected '(' after function name '{function_name_token.value}'")
        
        # Parse parameter list if not empty
        if self.current_token().type != TokenType.RIGHT_PAREN:
            while True:
                param_token = self.expect(TokenType.IDENTIFIER, "Expected parameter name")
                param_node = ASTNode(ASTNodeType.FUNCTION_PARAM, param_token.value, param_token)
                function_node.add_child(param_node)
                
                if self.match(TokenType.COMMA):
                    continue  # More parameters
                else:
                    break  # End of parameter list
        
        self.expect(TokenType.RIGHT_PAREN, "Expected ')' after function parameters")
        
        # Parse function body
        function_node.add_child(self.parse_block())
        
        return function_node
    
    def parse_block(self):
        """Parse a block of statements."""
        block_node = ASTNode(ASTNodeType.BLOCK)
        
        # Expect opening brace
        self.expect(TokenType.LEFT_BRACE, "Expected '{' to start block")
        
        # Parse statements until closing brace
        while self.current_token().type != TokenType.RIGHT_BRACE:
            if self.current_token().type == TokenType.EOF:
                raise ParserError("Unexpected end of file in block", self.current_token())
            
            try:
                statement = self.parse_statement()
                block_node.add_child(statement)
            except ParserError as e:
                # Skip to the next statement or the end of the block
                while (self.current_token().type not in [
                    TokenType.IF, TokenType.WHILE, TokenType.RETURN, 
                    TokenType.IDENTIFIER, TokenType.RIGHT_BRACE, TokenType.EOF
                ]):
                    self.advance()
        
        # Expect closing brace
        self.expect(TokenType.RIGHT_BRACE, "Expected '}' to close block")
        
        return block_node
    
    def parse_statement(self):
        """Parse a statement."""
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_statement()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        else:
            # For any other statement, assume it might be an assignment or expression
            # Get the current position to restore in case we need to rewind
            current_position = self.current
            current_token = self.current_token()
            
            # Try to parse as an assignment
            if current_token.type == TokenType.IDENTIFIER:
                # Look ahead to see if this is likely an assignment
                peek_token = self.peek()
                if peek_token.type == TokenType.ASSIGN:
                    # This is an assignment statement
                    id_token = self.advance()  # Consume the identifier
                    id_node = ASTNode(ASTNodeType.IDENTIFIER, id_token.value, id_token)
                    
                    self.advance()  # Consume the '=' token
                    
                    # Create an assignment node
                    assign_node = ASTNode(ASTNodeType.ASSIGN_STMT, token=id_token)
                    assign_node.add_child(id_node)
                    
                    # Parse the right-hand side expression
                    right = self.parse_expression()
                    assign_node.add_child(right)
                    
                    return assign_node
            
            # Reset to try other parsing methods if not an assignment
            self.current = current_position
            
            # Now try with the standard expression parser
            return self.parse_assignment_or_expression()
    
    def parse_if_statement(self):
        """Parse an if statement."""
        if_token = self.tokens[self.current - 1]  # Token already consumed by match()
        if_node = ASTNode(ASTNodeType.IF_STMT, token=if_token)
        
        # Parse condition
        condition = self.parse_expression()
        if_node.add_child(condition)
        
        # Parse then block
        then_block = self.parse_block()
        if_node.add_child(then_block)
        
        # Check for else clause
        if self.match(TokenType.ELSE):
            # Check if there's an "else if" pattern
            if self.match(TokenType.IF):
                # We have an "else if" - parse it as a nested if inside the else block
                else_block = ASTNode(ASTNodeType.BLOCK)  # Create an implicit block for the else
                
                # Parse the nested "if" inside the else block
                nested_if = self.parse_if_statement()  # This will handle the whole "if (...) {...}" including any chained "else if"s
                else_block.add_child(nested_if)
                
                if_node.add_child(else_block)
            else:
                # Regular else block
                else_block = self.parse_block()
                if_node.add_child(else_block)
        
        return if_node
    
    def parse_while_statement(self):
        """Parse a while statement."""
        while_token = self.tokens[self.current - 1]  # Token already consumed by match()
        while_node = ASTNode(ASTNodeType.WHILE_STMT, token=while_token)
        
        # Parse condition
        condition = self.parse_expression()
        while_node.add_child(condition)
        
        # Parse body
        body = self.parse_block()
        while_node.add_child(body)
        
        return while_node
    
    def parse_return_statement(self):
        """Parse a return statement."""
        return_token = self.tokens[self.current - 1]  # Token already consumed by match()
        return_node = ASTNode(ASTNodeType.RETURN_STMT, token=return_token)
        
        # Parse return value if present
        if self.current_token().type not in [TokenType.RIGHT_BRACE, TokenType.EOF]:
            expr = self.parse_expression()
            return_node.add_child(expr)
        
        return return_node
    
    def parse_assignment_or_expression(self):
        """Parse an assignment or expression statement."""
        expr = self.parse_expression()
        
        # If it's an assignment, wrap it in an assignment node
        if isinstance(expr, ASTNode) and expr.type == ASTNodeType.ASSIGN_STMT:
            return expr
        
        # Otherwise just return the expression
        return expr
    
    def parse_expression(self):
        """Parse an expression."""
        return self.parse_conditional()
    
    def parse_conditional(self):
        """Parse a conditional (ternary) expression."""
        expr = self.parse_logical_or()
        
        # Check for ternary operator: condition ? expr1 : expr2
        if self.match(TokenType.QUESTION):
            cond_node = ASTNode(ASTNodeType.CONDITIONAL_EXPR)
            cond_node.add_child(expr)  # Condition
            
            # Parse the 'then' expression
            then_expr = self.parse_expression()
            cond_node.add_child(then_expr)
            
            # Expect the colon
            self.expect(TokenType.COLON, "Expected ':' in conditional expression")
            
            # Parse the 'else' expression
            else_expr = self.parse_conditional()
            cond_node.add_child(else_expr)
            
            return cond_node
        
        return expr
    
    def parse_logical_or(self):
        """Parse logical OR expressions."""
        expr = self.parse_logical_and()
        
        while self.match(TokenType.OR):
            operator = self.tokens[self.current - 1]
            right = self.parse_logical_and()
            
            or_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            or_node.add_child(expr)
            or_node.add_child(right)
            expr = or_node
        
        return expr
    
    def parse_logical_and(self):
        """Parse logical AND expressions."""
        expr = self.parse_equality()
        
        while self.match(TokenType.AND):
                    operator = self.tokens[self.current - 1]
                    right = self.parse_equality()
                    
                    and_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
                    and_node.add_child(expr)
                    and_node.add_child(right)
                    expr = and_node
                
        return expr
    
    def parse_equality(self):
        """Parse equality expressions."""
        expr = self.parse_comparison()
        
        while self.match(TokenType.EQUALS, TokenType.NOT_EQUALS):
            operator = self.tokens[self.current - 1]
            right = self.parse_comparison()
            
            equal_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            equal_node.add_child(expr)
            equal_node.add_child(right)
            expr = equal_node
        
        return expr
    
    def parse_comparison(self):
        """Parse comparison expressions."""
        expr = self.parse_bitwise_or()
        
        while self.match(TokenType.LESS_THAN, TokenType.GREATER_THAN, 
                         TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL):
            operator = self.tokens[self.current - 1]
            right = self.parse_bitwise_or()
            
            comp_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            comp_node.add_child(expr)
            comp_node.add_child(right)
            expr = comp_node
        
        return expr
    
    def parse_bitwise_or(self):
        """Parse bitwise OR expressions."""
        expr = self.parse_bitwise_xor()
        
        while self.match(TokenType.BIT_OR):
            operator = self.tokens[self.current - 1]
            right = self.parse_bitwise_xor()
            
            or_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            or_node.add_child(expr)
            or_node.add_child(right)
            expr = or_node
        
        return expr
    
    def parse_bitwise_xor(self):
        """Parse bitwise XOR expressions."""
        expr = self.parse_bitwise_and()
        
        while self.match(TokenType.BIT_XOR):
            operator = self.tokens[self.current - 1]
            right = self.parse_bitwise_and()
            
            xor_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            xor_node.add_child(expr)
            xor_node.add_child(right)
            expr = xor_node
        
        return expr
    
    def parse_bitwise_and(self):
        """Parse bitwise AND expressions."""
        expr = self.parse_shift()
        
        while self.match(TokenType.BIT_AND):
            operator = self.tokens[self.current - 1]
            right = self.parse_shift()
            
            and_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            and_node.add_child(expr)
            and_node.add_child(right)
            expr = and_node
        
        return expr
    
    def parse_shift(self):
        """Parse bit shift expressions."""
        expr = self.parse_term()
        
        while self.match(TokenType.BIT_LSHIFT, TokenType.BIT_RSHIFT):
            operator = self.tokens[self.current - 1]
            right = self.parse_term()
            
            shift_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            shift_node.add_child(expr)
            shift_node.add_child(right)
            expr = shift_node
        
        return expr
    
    def parse_term(self):
        """Parse term expressions (+ and -)."""
        expr = self.parse_factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.tokens[self.current - 1]
            right = self.parse_factor()
            
            term_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            term_node.add_child(expr)
            term_node.add_child(right)
            expr = term_node
        
        return expr
    
    def parse_factor(self):
        """Parse factor expressions (* and /)."""
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MOD):
            operator = self.tokens[self.current - 1]
            right = self.parse_unary()
            
            factor_node = ASTNode(ASTNodeType.BINARY_EXPR, operator.value, operator)
            factor_node.add_child(expr)
            factor_node.add_child(right)
            expr = factor_node
        
        return expr
    
    def parse_unary(self):
        """Parse unary expressions."""
        if self.match(TokenType.NOT, TokenType.MINUS, TokenType.BIT_NOT):
            operator = self.tokens[self.current - 1]
            expr = self.parse_unary()
            
            unary_node = ASTNode(ASTNodeType.UNARY_EXPR, operator.value, operator)
            unary_node.add_child(expr)
            return unary_node
        
        return self.parse_postfix()
    
    def parse_postfix(self):
        """Parse postfix expressions (function calls and array accesses)."""
        expr = self.parse_primary()
        
        # Handle function calls and array accesses
        while True:
            if self.match(TokenType.LEFT_PAREN):
                # Function call
                call_node = ASTNode(ASTNodeType.FUNCTION_CALL, expr.value, expr.token)
                
                # Parse arguments
                if self.current_token().type != TokenType.RIGHT_PAREN:
                    while True:
                        arg = self.parse_expression()
                        call_node.add_child(arg)
                        
                        if self.match(TokenType.COMMA):
                            continue  # More arguments
                        else:
                            break  # End of argument list
                
                self.expect(TokenType.RIGHT_PAREN, "Expected ')' after function arguments")
                expr = call_node
            
            elif self.match(TokenType.LEFT_BRACKET):
                # Array access
                access_node = ASTNode(ASTNodeType.ARRAY_ACCESS, expr.value, expr.token)
                
                # Parse index expression
                index_expr = self.parse_expression()
                access_node.add_child(index_expr)
                
                self.expect(TokenType.RIGHT_BRACKET, "Expected ']' after array index")
                expr = access_node
            
            else:
                break  # No more postfix operators
        
        return expr
    
    def parse_primary(self):
        """Parse primary expressions."""
        # Literal values
        if self.match(TokenType.NUMBER):
            token = self.tokens[self.current - 1]
            return ASTNode(ASTNodeType.LITERAL, token.value, token)
        
        # Identifiers
        if self.match(TokenType.IDENTIFIER):
            token = self.tokens[self.current - 1]
            return ASTNode(ASTNodeType.IDENTIFIER, token.value, token)
        
        # Parenthesized expressions
        if self.match(TokenType.LEFT_PAREN):
            expr = self.parse_expression()
            self.expect(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        
        # This path is redundant with the improved parse_statement method, but kept for backward compatibility
        # Assignment
        if self.current_token().type == TokenType.IDENTIFIER:
            id_token = self.advance()
            id_node = ASTNode(ASTNodeType.IDENTIFIER, id_token.value, id_token)
            
            # Array access
            if self.match(TokenType.LEFT_BRACKET):
                access_node = ASTNode(ASTNodeType.ARRAY_ACCESS, id_token.value, id_token)
                
                # Parse index expression
                index_expr = self.parse_expression()
                access_node.add_child(index_expr)
                
                self.expect(TokenType.RIGHT_BRACKET, "Expected ']' after array index")
                
                # Check for assignment
                if self.match(TokenType.ASSIGN):
                    assign_node = ASTNode(ASTNodeType.ASSIGN_STMT, token=id_token)
                    assign_node.add_child(access_node)
                    
                    # Parse right-hand side
                    right = self.parse_expression()
                    assign_node.add_child(right)
                    
                    return assign_node
                
                return access_node
            
            # Simple assignment
            if self.match(TokenType.ASSIGN):
                assign_node = ASTNode(ASTNodeType.ASSIGN_STMT, token=id_token)
                assign_node.add_child(id_node)
                
                # Parse right-hand side
                right = self.parse_expression()
                assign_node.add_child(right)
                
                return assign_node
            
            return id_node
        
        raise ParserError(f"Unexpected token: {self.current_token().type.name}", self.current_token())


def parse(tokens):
    """Parse tokens into an AST."""
    parser = Parser(tokens)
    ast = parser.parse()
    return ast, parser.errors

def parse_tokens(tokens):
    """Parse tokens into an AST."""
    return parse(tokens)