import numpy as np

class PolyC:
    def __init__(self):
        self.position = np.zeros([3], dtype=float)  # Input
        self.velocity = np.zeros([3], dtype=float)  # Action
        self.maxSpeed = float(0)  # Constant
        self.currentSpeed = float(0)  # State variable
        self.goalPosition = np.zeros([3], dtype=float)  # Environment variable
        self.totalTime = float(0)  # Environment variable
        self.health = float(0)  # State variable

    def check_constraints(self):
        violations = []
        if not ((self.velocity < self.maxSpeed)):
            violations.append('Constraint 1 violated: (self.velocity < self.maxSpeed)')
        if not ((self.health > 0)):
            violations.append('Constraint 2 violated: (self.health > 0)')
        return violations

    def evaluate_goals(self):
        goal_values = {}
        goal_values['goal_1'] = {'type': 'equality', 'value': (self.position == self.goalPosition)}
        goal_values['goal_2'] = {'type': 'maximize', 'value': self.health}
        goal_values['goal_3'] = {'type': 'minimize', 'value': self.totalTime}
        return goal_values

    def execute_policy(self):
            self.velocity = 0
            if (self.velocity > self.maxSpeed):
                            self.velocity = self.maxSpeed

    def add(self, x, y):
            return (x + y)

