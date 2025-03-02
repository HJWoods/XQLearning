import numpy as np

class PolyC:
    def __init__(self):
        self.paddle_y = float(0)  # Input
        self.ball_pos = np.zeros([2], dtype=float)  # Input
        self.ball_vel = np.zeros([2], dtype=float)  # Input
        self.opp_paddle_y = float(0)  # Input
        self.score_diff = float(0)  # Input
        self.paddle_move = float(0)  # Action
        self.paddle_height = 0.2  # Constant
        self.court_height = 1.0  # Constant
        self.max_speed = 1.0  # Constant
        self.true_ball_dir = float(0)  # Environment variable
        self.predicted_y = float(0)  # State variable
        self.target_y = float(0)  # State variable
        self.last_score_diff = float(0)  # State variable
        self.time_to_reach = float(0)  # State variable
        self.y_pos = float(0)  # State variable

    def check_constraints(self):
        violations = []
        if not ((self.paddle_move >= (-1.0))):
            violations.append('Constraint 1 violated: (self.paddle_move >= (-1.0))')
        if not ((self.paddle_move <= 1.0)):
            violations.append('Constraint 2 violated: (self.paddle_move <= 1.0)')
        return violations

    def evaluate_goals(self):
        goal_values = {}
        goal_values['goal_1'] = {'type': 'maximize', 'value': self.score_diff}
        goal_values['goal_2'] = {'type': 'minimize', 'value': self.abs((self.paddle_y - self.predicted_y))}
        return goal_values

    def execute_policy(self):
            if (self.score_diff != self.last_score_diff):
                            self.last_score_diff = self.score_diff
            self.predicted_y = self.predict_intersection()
            self.target_y = (self.predicted_y - (self.paddle_height / 2))
            if (self.target_y < 0):
                            self.target_y = 0
            if (self.target_y > (self.court_height - self.paddle_height)):
                            self.target_y = (self.court_height - self.paddle_height)
            if (self.paddle_y > (self.target_y + 0.02)):
                            self.paddle_move = (-self.max_speed)
            else:
                            if (self.paddle_y < (self.target_y - 0.02)):
                                                    self.paddle_move = self.max_speed
                            else:
                                                    self.paddle_move = 0

    def abs(self, x):
            if (x < 0):
                            return (-x)
            return x

    def predict_intersection(self, ):
            if (self.ball_vel[0] >= 0):
                            return self.paddle_y
            self.time_to_reach = ((-self.ball_pos[0]) / self.ball_vel[0])
            self.y_pos = (self.ball_pos[1] + (self.ball_vel[1] * self.time_to_reach))
            while ((self.y_pos < 0) or (self.y_pos > self.court_height)):
                            if (self.y_pos < 0):
                                                    self.y_pos = (-self.y_pos)
                            if (self.y_pos > self.court_height):
                                                    self.y_pos = ((2 * self.court_height) - self.y_pos)
            return self.y_pos

