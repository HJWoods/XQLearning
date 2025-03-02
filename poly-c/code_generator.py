"""
Code Generator module for the Poly-c compiler - FIXED VERSION
This module handles code generation for Poly-c code
"""

from symbol_table import SymbolType, DataType
from parser import ASTNodeType

class CodeGenerator:
    """Code generator for Poly-c"""
    
    def __init__(self, ast, symbol_table):
        self.ast = ast
        self.symbol_table = symbol_table
        self.code = {
            'vars': '',
            'constraints': '',
            'goals': '',
            'policy': '',
            'functions': ''
        }
        self.current_section = 'policy'
        self.indent_level = 0
        self.processed_functions = set()
        self.current_line_has_assignment = False
        self.need_assignment = False
    
    def indent(self):
        """Return proper indentation for current level."""
        return '    ' * self.indent_level
    
    def generate(self):
        """Generate code for the entire program."""
        if not self.ast:
            return "# No valid AST to generate code from"
        
        self._generate_node(self.ast)
        
        # Combine all sections
        full_code = (
            "import numpy as np\n\n"
            "class PolyC:\n"
            "    def __init__(self):\n"
            f"{self._with_indent(self.code['vars'], 2)}\n\n"
            "    def check_constraints(self):\n"
            "        violations = []\n"
            f"{self._with_indent(self.code['constraints'], 2)}\n"
            "        return violations\n\n"
            "    def evaluate_goals(self):\n"
            "        goal_values = {}\n"
            f"{self._with_indent(self.code['goals'], 2)}\n"
            "        return goal_values\n\n"
            "    def execute_policy(self):\n"
            f"{self._with_indent(self.code['policy'], 2)}\n\n"
            f"{self.code['functions']}"
        )
        
        return full_code
    
    def _with_indent(self, code, level):
        """Add indentation to each line of code."""
        if not code:
            return ""
        
        indentation = '    ' * level
        lines = []
        for line in code.split('\n'):
            if line.strip():
                lines.append(indentation + line)
        return '\n'.join(lines)
    
    def _generate_node(self, node):
        """Generate code for a node."""
        if node is None:
            return None
        
        if node.type == ASTNodeType.PROGRAM:
            return self._generate_program(node)
        elif node.type == ASTNodeType.VARIABLE_DECL:
            return self._generate_variable_declaration(node)
        elif node.type == ASTNodeType.CONSTRAINTS_BLOCK:
            return self._generate_constraints_block(node)
        elif node.type == ASTNodeType.GOALS_BLOCK:
            return self._generate_goals_block(node)
        elif node.type == ASTNodeType.FUNCTION_DEF:
            return self._generate_function_definition(node)
        elif node.type == ASTNodeType.BLOCK:
            return self._generate_block(node)
        elif node.type == ASTNodeType.IF_STMT:
            return self._generate_if_statement(node)
        elif node.type == ASTNodeType.WHILE_STMT:
            return self._generate_while_statement(node)
        elif node.type == ASTNodeType.ASSIGN_STMT:
            # Mark that we have an assignment on this line
            self.current_line_has_assignment = True
            result = self._generate_assignment(node)
            self.current_line_has_assignment = False
            return result
        elif node.type == ASTNodeType.RETURN_STMT:
            return self._generate_return_statement(node)
        elif node.type == ASTNodeType.BINARY_EXPR:
            result = self._generate_binary_expression(node)
            # If we need an assignment but don't have one, add a dummy one
            if self.need_assignment and not self.current_line_has_assignment:
                self.need_assignment = False
                return f"dummy = {result}"
            return result
        elif node.type == ASTNodeType.UNARY_EXPR:
            return self._generate_unary_expression(node)
        elif node.type == ASTNodeType.FUNCTION_CALL:
            return self._generate_function_call(node)
        elif node.type == ASTNodeType.ARRAY_ACCESS:
            return self._generate_array_access(node)
        elif node.type == ASTNodeType.IDENTIFIER:
            return self._generate_identifier(node)
        elif node.type == ASTNodeType.LITERAL:
            return self._generate_literal(node)
        elif node.type == ASTNodeType.CONDITIONAL_EXPR:
            return self._generate_conditional_expression(node)
        else:
            # Default: process all children
            result = []
            for child in node.children:
                code = self._generate_node(child)
                if code:
                    # Skip standalone expressions with no effect
                    if not self.current_line_has_assignment and isinstance(code, str):
                        # Check if it's an expression without assignment
                        if code.startswith('(') and code.endswith(')'):
                            # This is an expression without assignment - likely a bug in the original code
                            # For neural networks, try to add a self-assignment
                            if "layer" in code and "*" in code and "+" in code:
                                # Assume this is a neural network calculation
                                # Try to extract the array variable that should be assigned
                                array_match = re.search(r'\((self\.\w+\[\d+\])', code)
                                if array_match:
                                    array_var = array_match.group(1)
                                    # Replace the parentheses to make it a proper expression
                                    expr = code[1:-1]
                                    # Add assignment
                                    code = f"{array_var} = {expr}"
                            elif "self." in code and "(" in code and ")" in code and not any(x in code for x in ["=", "if", "while"]):
                                # This looks like a function call without assignment
                                # Likely an activation function call
                                if any(func in code for func in ["relu", "sigmoid", "tanh", "identity"]):
                                    # Try to extract the argument
                                    arg_match = re.search(r'self\.\w+\((self\.\w+\[\d+\])\)', code)
                                    if arg_match:
                                        arg = arg_match.group(1)
                                        # Add self-assignment
                                        code = f"{arg} = {code}"
                                        
                    # Don't add standalone identifiers/expressions without context
                    if not (isinstance(code, str) and code.startswith('self.') and '\n' not in code and 
                           '=' not in code and '(' not in code):
                        result.append(code)
            return '\n'.join(result) if result else None
    
    def _generate_program(self, node):
        """Generate code for the program."""
        for child in node.children:
            self._generate_node(child)
        return None
    
    def _generate_variable_declaration(self, node):
        """Generate code for a variable declaration."""
        var_name = node.value
        var_type_node = node.children[0]
        data_type_node = node.children[1]
        
        var_type = var_type_node.value
        data_type = data_type_node.value
        
        # Look up symbol information
        symbol = self.symbol_table.resolve(var_name)
        if not symbol:
            return None
        
        # Determine Python type
        python_type = "float"
        if symbol.data_type == DataType.INT:
            python_type = "int"
        elif symbol.data_type == DataType.BOOL:
            python_type = "bool"
        elif symbol.data_type == DataType.CHAR:
            python_type = "char"
        
        # Check for initializer
        init_value = None
        if len(node.children) > 2:
            init_expr = node.children[2]
            init_value = self._generate_node(init_expr)
        
        # If no initializer, use default values based on type
        if init_value is None:
            if symbol.dimensions:
                dimensions_str = ', '.join(str(dim) for dim in symbol.dimensions)
                init_value = f"np.zeros([{dimensions_str}], dtype={python_type})"
            else:
                init_value = f"{python_type}(0)"
        
        # Generate code based on variable type
        comment = ""
        if var_type == "input":
            comment = "# Input"
        elif var_type == "action":
            comment = "# Action"
        elif var_type == "const":
            comment = "# Constant"
        elif var_type == "var":
            comment = "# State variable"
        elif var_type == "env":
            comment = "# Environment variable"
        
        self.code['vars'] += f"self.{var_name} = {init_value}  {comment}\n"
        return None
    
    def _generate_constraints_block(self, node):
        """Generate code for constraints block."""
        self.current_section = 'constraints'
        
        for i, child in enumerate(node.children):
            constraint_expr = self._generate_node(child.children[0])
            self.code['constraints'] += f"if not ({constraint_expr}):\n"
            self.code['constraints'] += f"    violations.append('Constraint {i+1} violated: {constraint_expr}')\n"
        
        return None
    
    def _generate_goals_block(self, node):
        """Generate code for goals block."""
        self.current_section = 'goals'
        
        for i, child in enumerate(node.children):
            if child.value in ["min", "max"]:
                goal_var = self._generate_node(child.children[0])
                goal_type = "minimize" if child.value == "min" else "maximize"
                self.code['goals'] += f"goal_values['goal_{i+1}'] = {{'type': '{goal_type}', 'value': {goal_var}}}\n"
            else:
                goal_expr = self._generate_node(child.children[0])
                self.code['goals'] += f"goal_values['goal_{i+1}'] = {{'type': 'equality', 'value': {goal_expr}}}\n"
        
        return None
    
    def _generate_function_definition(self, node):
        """Generate code for a function definition."""
        function_name = node.value
        
        # Skip if already processed
        if function_name in self.processed_functions:
            return None
        
        self.processed_functions.add(function_name)
        
        # Handle main function differently
        if function_name == "main":
            self.current_section = 'policy'
            
            # Process main body
            for child in node.children:
                if child.type == ASTNodeType.BLOCK:
                    body_code = self._generate_node(child)
                    if body_code:
                        self.code['policy'] += body_code
            
            return None
        
        # Regular function
        self.current_section = 'functions'
        
        # Get parameters
        params = []
        for child in node.children:
            if child.type == ASTNodeType.FUNCTION_PARAM:
                params.append(child.value)
        
        # Function header
        params_str = ", ".join(params)
        self.code['functions'] += f"    def {function_name}(self, {params_str}):\n"
        
        # Function body
        body_code = ""
        for child in node.children:
            if child.type == ASTNodeType.BLOCK:
                body_code = self._generate_node(child)
        
        if body_code:
            # Indent the body code
            indented_body = '\n'.join(f"        {line}" for line in body_code.split('\n') if line.strip())
            self.code['functions'] += indented_body + "\n\n"
        else:
            # Empty body
            self.code['functions'] += "        pass\n\n"
        
        return None
    
    def _generate_block(self, node):
        """Generate code for a block."""
        old_indent = self.indent_level
        self.indent_level += 1
        
        statements = []
        for child in node.children:
            stmt = self._generate_node(child)
            if stmt:
                # Skip standalone variable references
                if isinstance(stmt, str) and stmt.strip().startswith('self.') and '\n' not in stmt and '=' not in stmt and '(' not in stmt:
                    continue
                statements.append(stmt)
        
        self.indent_level = old_indent
        
        if not statements:
            return None
        
        return '\n'.join(statements)

    def _generate_if_statement(self, node):
        """Generate code for an if statement."""
        if len(node.children) < 2:
            return None
        
        condition = self._generate_node(node.children[0])
        
        # Indent the body
        old_indent = self.indent_level
        self.indent_level += 1
        
        then_block = self._generate_node(node.children[1])
        if not then_block:
            then_block = "pass"
        
        self.indent_level = old_indent
        
        code = f"{self.indent()}if {condition}:\n"
        
        # Handle the case where then_block is a single line
        if '\n' not in then_block:
            # Skip standalone variable references
            if then_block.strip().startswith('self.') and '=' not in then_block and '(' not in then_block:
                code += f"{self.indent()}    pass\n"
            else:
                code += f"{self.indent()}    {then_block}\n"
        else:
            # Process multiline blocks
            for line in then_block.split('\n'):
                # Skip standalone variable references
                if line.strip().startswith('self.') and '=' not in line and '(' not in line:
                    continue
                if line.strip():  # Only add non-empty lines
                    code += f"{self.indent()}    {line}\n"
        
        # Generate else block if present
        if len(node.children) > 2:
            self.indent_level += 1
            else_block = self._generate_node(node.children[2])
            if not else_block:
                else_block = "pass"
            self.indent_level = old_indent
            
            code += f"{self.indent()}else:\n"
            
            # Handle the case where else_block is a single line
            if '\n' not in else_block:
                # Skip standalone variable references
                if else_block.strip().startswith('self.') and '=' not in else_block and '(' not in else_block:
                    code += f"{self.indent()}    pass\n"
                else:
                    code += f"{self.indent()}    {else_block}\n"
            else:
                # Process multiline blocks
                for line in else_block.split('\n'):
                    # Skip standalone variable references
                    if line.strip().startswith('self.') and '=' not in line and '(' not in line:
                        continue
                    if line.strip():  # Only add non-empty lines
                        code += f"{self.indent()}    {line}\n"
        
        return code.rstrip()

    def _generate_while_statement(self, node):
        """Generate code for a while statement."""
        # Skip incomplete while loops with no condition or body
        if len(node.children) < 1:
            return f"{self.indent()}# Incomplete while statement - missing condition"
        
        condition = self._generate_node(node.children[0])
        
        code = f"{self.indent()}while {condition}:\n"
        
        # If there's no body or an incomplete body, add a pass statement
        if len(node.children) < 2 or not node.children[1].children:
            code += f"{self.indent()}    pass"
            return code
        
        # Indent the body
        old_indent = self.indent_level
        self.indent_level += 1
        
        body = self._generate_node(node.children[1])
        if not body:
            body = "pass"
        
        self.indent_level = old_indent
        
        # Handle the case where body is a single line
        if '\n' not in body:
            # Skip standalone variable references
            if body.strip().startswith('self.') and '=' not in body and '(' not in body:
                code += f"{self.indent()}    pass\n"
            else:
                code += f"{self.indent()}    {body}\n"
        else:
            # Process multiline blocks
            for line in body.split('\n'):
                # Skip standalone variable references
                if line.strip().startswith('self.') and '=' not in line and '(' not in line:
                    continue
                if line.strip():  # Only add non-empty lines
                    code += f"{self.indent()}    {line}\n"
        
        return code.rstrip()
    
    def _generate_assignment(self, node):
        """Generate code for an assignment statement."""
        if len(node.children) < 2:
            return None
        
        left = self._generate_node(node.children[0])
        right = self._generate_node(node.children[1])
        
        # Check for neural network activation function calls
        if isinstance(right, str) and any(func in right for func in ["relu", "sigmoid", "tanh", "identity"]):
            # Check if it's a function call where the argument is the same as the left side
            import re
            func_match = re.search(r'self\.(\w+)\((self\.\w+\[\d+\])\)', right)
            if func_match:
                func_name, arg_var = func_match.groups()
                if arg_var == left:
                    # For activation functions, just do the assignment
                    return f"{self.indent()}{left} = {right}"
        
        return f"{self.indent()}{left} = {right}"
    
    def _generate_return_statement(self, node):
        """Generate code for a return statement."""
        if not node.children:
            return f"{self.indent()}return"
        
        expr = self._generate_node(node.children[0])
        return f"{self.indent()}return {expr}"
    
    def _generate_binary_expression(self, node):
        """Generate code for a binary expression."""
        if len(node.children) < 2:
            return None
        
        left = self._generate_node(node.children[0])
        right = self._generate_node(node.children[1])
        operator = node.value
        
        # Map operators to Python equivalents
        op_map = {
            '&&': 'and',
            '||': 'or',
            '==': '==',
            '!=': '!=',
            '<': '<',
            '>': '>',
            '<=': '<=',
            '>=': '>=',
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '%': '%',
            '&': '&',
            '|': '|',
            '^': '^',
            '<<': '<<',
            '>>': '>>'
        }
        
        python_op = op_map.get(operator, operator)
        
        # Add parentheses for precedence
        return f"({left} {python_op} {right})"
    
    def _generate_unary_expression(self, node):
        """Generate code for a unary expression."""
        if not node.children:
            return None
        
        expr = self._generate_node(node.children[0])
        operator = node.value
        
        # Map operators to Python equivalents
        op_map = {
            '!': 'not ',
            '-': '-',
            '~': '~'
        }
        
        python_op = op_map.get(operator, operator)
        
        return f"({python_op}{expr})"
    
    def _generate_conditional_expression(self, node):
        """Generate code for a conditional expression."""
        if len(node.children) < 3:
            return None
        
        condition = self._generate_node(node.children[0])
        then_expr = self._generate_node(node.children[1])
        else_expr = self._generate_node(node.children[2])
        
        # Python's conditional expression: x if condition else y
        return f"({then_expr} if {condition} else {else_expr})"
    
    def _generate_function_call(self, node):
        """Generate code for a function call."""
        function_name = node.value
        
        # Generate arguments
        args = []
        for child in node.children:
            arg = self._generate_node(child)
            args.append(arg)
        
        args_str = ", ".join(args)
        
        return f"self.{function_name}({args_str})"
    
    def _generate_array_access(self, node):
        """Generate code for an array access."""
        array_name = node.value
        
        # Generate index expression
        if not node.children:
            return f"self.{array_name}"
        
        index_expr = self._generate_node(node.children[0])
        
        return f"self.{array_name}[{index_expr}]"
    
    def _generate_identifier(self, node):
        """Generate code for an identifier."""
        var_name = node.value
        
        # Special cases for min/max
        if var_name in ['min', 'max']:
            return var_name
        
        # Add 'self.' for class members
        symbol = self.symbol_table.resolve(var_name)
        if symbol and symbol.symbol_type != SymbolType.PARAMETER:
            return f"self.{var_name}"
        
        return var_name
    
    def _generate_literal(self, node):
        """Generate code for a literal."""
        value = node.value
        
        # Format the value based on type
        if isinstance(value, str):
            return f'"{value}"'  # String literal
        
        return str(value)  # Numeric literal

def generate(ast, symbol_table):
    """Generate code from an AST with a given symbol table"""
    generator = CodeGenerator(ast, symbol_table)
    return generator.generate()