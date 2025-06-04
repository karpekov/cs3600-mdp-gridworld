"""
Grid World MDP Visualization Module
===================================
A comprehensive module for creating and visualizing grid world MDPs with
value iteration and policy iteration, featuring enhanced ipywidgets for
interactive exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import matplotlib.colors as mcolors

class GridWorldMDP:
    """
    A configurable Grid World MDP environment with visualization capabilities.
    """

    def __init__(self, rows=3, cols=4, gamma=0.9, noise=0.2, step_cost=0.0, formula_type='BERKELEY'):
        """
        Initialize the Grid World MDP.

        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            gamma (float): Discount factor
            noise (float): Probability of taking unintended actions (0.1 means 10% each side)
            step_cost (float): Cost/reward for each step (negative = cost, positive = reward)
            formula_type (str): 'AIMA' or 'BERKELEY' - determines reward calculation method
                - AIMA: Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))
                - BERKELEY: Q = R(s) + γ * Σ P(s'|s,a) * V(s')
        """
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.noise = noise
        self.step_cost = step_cost
        self.formula_type = formula_type.upper()

        if self.formula_type not in ['AIMA', 'BERKELEY']:
            raise ValueError("formula_type must be 'AIMA' or 'BERKELEY'")

        # Default terminals for 3x4 grid (can be customized)
        self.terminals = {(0, 3): +1, (1, 3): -1}
        self.obstacles = set()  # Default: no obstacles

        # Actions and transitions
        self.actions = ['U', 'D', 'L', 'R']
        self.delta = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.left_of = {'U': 'L', 'L': 'D', 'D': 'R', 'R': 'U'}
        self.right_of = {v: k for k, v in self.left_of.items()}

        # Visualization settings
        self.arrow_symbols = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}

        # Storage for iteration history
        self.reset_history()

    def set_terminals(self, terminals_dict):
        """Set terminal states and their rewards."""
        self.terminals = terminals_dict.copy()

    def set_obstacles(self, obstacles_set):
        """Set obstacle states."""
        if obstacles_set is None:
            self.obstacles = set()
        else:
            self.obstacles = obstacles_set.copy()

    def set_step_cost(self, step_cost):
        """Set the step cost/reward."""
        self.step_cost = step_cost

    def set_initial_policy(self, policy_dict):
        """
        Set a custom initial policy for policy iteration.

        Args:
            policy_dict (dict): Dictionary mapping states to actions
                               Format: {(row, col): action_string}
                               Example: {(0,0): 'R', (0,1): 'R', ...}
        """
        self.custom_initial_policy = {}

        # Validate and convert to internal format
        for state, action in policy_dict.items():
            if not self.is_valid_state(state[0], state[1]):
                continue
            if state in self.terminals:
                continue  # Skip terminal states
            if action not in self.actions:
                raise ValueError(f"Invalid action '{action}' for state {state}. Must be one of {self.actions}")

            # Store as list for consistency with policy representation
            self.custom_initial_policy[state] = [action]

        print(f"Custom initial policy set for {len(self.custom_initial_policy)} states")

    def create_directional_policy(self, direction):
        """
        Create a policy that always chooses the same direction.

        Args:
            direction (str): One of 'U', 'D', 'L', 'R', 'UP', 'DOWN', 'LEFT', 'RIGHT'

        Returns:
            dict: Policy dictionary suitable for set_initial_policy()
        """
        # Handle different direction formats
        direction_map = {
            'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R',
            'U': 'U', 'D': 'D', 'L': 'L', 'R': 'R'
        }

        if direction.upper() not in direction_map:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {list(direction_map.keys())}")

        action = direction_map[direction.upper()]
        policy = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if self.is_valid_state(r, c) and state not in self.terminals:
                    policy[state] = action

        return policy

    def create_random_policy(self, seed=None):
        """
        Create a random policy.

        Args:
            seed (int, optional): Random seed for reproducibility

        Returns:
            dict: Policy dictionary suitable for set_initial_policy()
        """
        if seed is not None:
            np.random.seed(seed)

        policy = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if self.is_valid_state(r, c) and state not in self.terminals:
                    action = np.random.choice(self.actions)
                    policy[state] = action

        return policy

    def create_clockwise_policy(self):
        """
        Create a policy that follows a clockwise pattern around the grid.

        Returns:
            dict: Policy dictionary suitable for set_initial_policy()
        """
        policy = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if not self.is_valid_state(r, c) or state in self.terminals:
                    continue

                # Clockwise movement: top row goes right, right column goes down,
                # bottom row goes left, left column goes up
                if r == 0:  # Top row
                    action = 'R'
                elif c == self.cols - 1:  # Right column
                    action = 'D'
                elif r == self.rows - 1:  # Bottom row
                    action = 'L'
                elif c == 0:  # Left column
                    action = 'U'
                else:  # Interior - head towards nearest edge clockwise
                    # Simple heuristic: head right if closer to top/bottom, down if closer to edges
                    if min(r, self.rows - 1 - r) <= min(c, self.cols - 1 - c):
                        action = 'R'  # Head toward right edge first
                    else:
                        action = 'D'  # Head toward bottom edge first

                policy[state] = action

        return policy

    def clear_initial_policy(self):
        """Clear any custom initial policy, reverting to random initialization."""
        if hasattr(self, 'custom_initial_policy'):
            del self.custom_initial_policy
        print("Custom initial policy cleared")

    def print_policy(self, policy_dict, title="Policy"):
        """
        Print a policy in a readable format.

        Args:
            policy_dict (dict): Policy to print (either custom format or internal format)
            title (str): Title for the policy display
        """
        print(f"\n{title}:")
        print("=" * (len(title) + 1))

        for r in range(self.rows):
            row_display = []
            for c in range(self.cols):
                state = (r, c)

                if state in self.terminals:
                    reward = self.terminals[state]
                    row_display.append(f"{reward:+2.0f}")
                elif state in self.obstacles:
                    row_display.append(" ■ ")
                elif state in policy_dict:
                    # Handle both custom format {state: action} and internal format {state: [action]}
                    actions = policy_dict[state]
                    if isinstance(actions, list):
                        if len(actions) == 1:
                            row_display.append(f" {actions[0]} ")
                        else:
                            row_display.append(f"{len(actions)}")  # Show number of tied actions
                    else:
                        row_display.append(f" {actions} ")
                else:
                    row_display.append(" ? ")

            print(" ".join(row_display))
        print()

    def reset_history(self):
        """Reset all iteration history."""
        self.v_history = []
        self.q_history = []
        self.formula_history = []
        self.policy_history = []
        self.converged_iteration = None

        # Reset policy iteration specific history
        self.policy_eval_history = []
        self.policy_eval_v_history = []

    def in_grid(self, r, c):
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_valid_state(self, r, c):
        """Check if state is valid (in grid and not an obstacle)."""
        return self.in_grid(r, c) and (r, c) not in self.obstacles

    def get_transitions(self, state, action):
        """
        Get transition probabilities for a given state-action pair.
        Returns list of (probability, next_state, reward) tuples.

        For AIMA: reward includes both current step cost and terminal reward
        For BERKELEY: reward is only the terminal reward (step cost handled separately)
        """
        if state in self.terminals:
            return [(1.0, state, 0)]  # Terminal states are absorbing

        # Calculate intended and unintended actions
        # noise represents total slip probability, split equally between left and right
        intended_prob = 1.0 - self.noise
        side_prob = self.noise / 2.0

        transitions = []
        actions_to_try = [action, self.left_of[action], self.right_of[action]]
        probabilities = [intended_prob, side_prob, side_prob]

        for prob, act in zip(probabilities, actions_to_try):
            dr, dc = self.delta[act]
            nr, nc = state[0] + dr, state[1] + dc

            # Check if new position is valid
            if not self.is_valid_state(nr, nc):
                nr, nc = state  # Stay in same place if invalid

            # Calculate reward based on formula type
            terminal_reward = self.terminals.get((nr, nc), 0)

            if self.formula_type == 'AIMA':
                # AIMA: reward for transitioning TO next state
                # - Terminal reward if transitioning to terminal state
                # - Step cost if transitioning to non-terminal state
                if (nr, nc) in self.terminals:
                    total_reward = terminal_reward
                else:
                    total_reward = self.step_cost
            else:  # BERKELEY
                # BERKELEY: only terminal reward in transition, step cost handled separately
                total_reward = terminal_reward

            transitions.append((prob, (nr, nc), total_reward))

        return transitions

    def compute_q_values_and_formulas(self, V):
        """
        Compute Q-values and their formula representations for all state-action pairs.

        Returns:
            q_dict: {state: {action: q_value}}
            formula_dict: {state: {action: formula_string}}
        """
        q_values = {}
        formulas = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)

                if state in self.terminals or state in self.obstacles:
                    continue

                q_values[state] = {}
                formulas[state] = {}

                for action in self.actions:
                    transitions = self.get_transitions(state, action)

                    if self.formula_type == 'AIMA':
                        # AIMA: Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))
                        q_val = 0.0
                        formula_parts = []

                        for prob, next_state, reward in transitions:
                            term_value = reward + self.gamma * V[next_state]
                            q_val += prob * term_value
                            formula_parts.append(f'{prob:.1f}*({reward:+.1f} + {self.gamma:.1f}*{V[next_state]:.3f})')

                        formula_str = " + ".join(formula_parts) + f' = {q_val:.3f}'
                        value = q_val

                    else:  # BERKELEY
                        # BERKELEY: Q = R(s) + γ * Σ P(s'|s,a) * V(s')
                        # R(s) is step cost only for non-terminal transitions
                        # Terminal states already have their values set correctly in V
                        step_cost_contribution = 0.0
                        expected_next_value = 0.0
                        prob_parts = []

                        for prob, next_state, reward in transitions:
                            # Only pay step cost if not transitioning to terminal state
                            if next_state not in self.terminals:
                                step_cost_contribution += prob * self.step_cost
                            expected_next_value += prob * V[next_state]
                            prob_parts.append(f'{prob:.1f}*{V[next_state]:.3f}')

                        value = step_cost_contribution + self.gamma * expected_next_value

                        if step_cost_contribution != 0:
                            formula_str = f'{step_cost_contribution:+.3f} + {self.gamma:.1f}*({" + ".join(prob_parts)}) = {value:.3f}'
                        else:
                            formula_str = f'{self.gamma:.1f}*({" + ".join(prob_parts)}) = {value:.3f}'

                    q_values[state][action] = value
                    formulas[state][action] = formula_str

        return q_values, formulas

    def get_greedy_policy(self, V, tolerance=1e-6):
        """Extract greedy policy from value function. Returns all tied optimal actions."""
        policy = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)

                if state in self.terminals or state in self.obstacles:
                    continue

                action_values = {}

                for action in self.actions:
                    transitions = self.get_transitions(state, action)

                    if self.formula_type == 'AIMA':
                        # AIMA: Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))
                        q_value = 0.0
                        for prob, next_state, reward in transitions:
                            q_value += prob * (reward + self.gamma * V[next_state])
                    else:  # BERKELEY
                        # BERKELEY: Q = R(s) + γ * Σ P(s'|s,a) * V(s')
                        step_cost_contribution = 0.0
                        expected_next_value = 0.0

                        for prob, next_state, reward in transitions:
                            # Only pay step cost if not transitioning to terminal state
                            if next_state not in self.terminals:
                                step_cost_contribution += prob * self.step_cost
                            expected_next_value += prob * V[next_state]

                        q_value = step_cost_contribution + self.gamma * expected_next_value

                    action_values[action] = q_value

                # Find all actions with maximum value (within tolerance)
                max_value = max(action_values.values())
                optimal_actions = [action for action, value in action_values.items()
                                 if abs(value - max_value) <= tolerance]

                policy[state] = optimal_actions

        return policy

    def value_iteration(self, max_iterations=100, tolerance=1e-6):
        """
        Run value iteration algorithm with full history tracking.

        Returns:
            converged_iteration: The iteration where convergence occurred
        """
        self.reset_history()

        # Initialize value function
        V = np.zeros((self.rows, self.cols))

        # Initialize terminal states based on formulation
        if self.formula_type == 'BERKELEY':
            # BERKELEY: terminal states have their reward values
            for state, reward in self.terminals.items():
                V[state] = reward
        # AIMA: terminal states remain at 0 (rewards come from transitions TO them)

        # Store initial state
        q_vals, formulas = self.compute_q_values_and_formulas(V)
        policy = self.get_greedy_policy(V)

        self.v_history.append(V.copy())
        self.q_history.append(q_vals)
        self.formula_history.append(formulas)
        self.policy_history.append(policy)

        for iteration in range(max_iterations):
            delta = 0.0
            new_V = V.copy()

            # Update all non-terminal, non-obstacle states
            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)

                    if state in self.terminals or state in self.obstacles:
                        continue

                    # Find best action value
                    action_values = []
                    for action in self.actions:
                        transitions = self.get_transitions(state, action)

                        if self.formula_type == 'AIMA':
                            # AIMA: Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))
                            q_value = 0.0
                            for prob, next_state, reward in transitions:
                                q_value += prob * (reward + self.gamma * V[next_state])
                        else:  # BERKELEY
                            # BERKELEY: Q = R(s) + γ * Σ P(s'|s,a) * V(s')
                            step_cost_contribution = 0.0
                            expected_next_value = 0.0

                            for prob, next_state, reward in transitions:
                                # Only pay step cost if not transitioning to terminal state
                                if next_state not in self.terminals:
                                    step_cost_contribution += prob * self.step_cost
                                expected_next_value += prob * V[next_state]

                            q_value = step_cost_contribution + self.gamma * expected_next_value

                        action_values.append(q_value)

                    best_value = max(action_values)
                    delta = max(delta, abs(best_value - V[state]))
                    new_V[state] = best_value

            V = new_V

            # Keep terminal states at correct values based on formulation
            if self.formula_type == 'BERKELEY':
                # BERKELEY: maintain terminal state reward values
                for state, reward in self.terminals.items():
                    V[state] = reward
            # AIMA: terminal states remain at 0

            # Store iteration results
            q_vals, formulas = self.compute_q_values_and_formulas(V)
            policy = self.get_greedy_policy(V)

            self.v_history.append(V.copy())
            self.q_history.append(q_vals)
            self.formula_history.append(formulas)
            self.policy_history.append(policy)

            # Check convergence
            if delta < tolerance:
                self.converged_iteration = iteration + 1
                break

        return self.converged_iteration or max_iterations

    def policy_iteration(self, max_iterations=50):
        """
        Run policy iteration algorithm with full history tracking.
        Uses custom initial policy if set via set_initial_policy(), otherwise random.

        Returns:
            converged_iteration: The iteration where convergence occurred
        """
        self.reset_history()

        # Initialize storage for policy evaluation steps
        self.policy_eval_history = []  # List of lists: [policy_iter][eval_step]
        self.policy_eval_v_history = []  # List of lists: [policy_iter][eval_step]

        # Initialize policy - use custom if available, otherwise random
        if hasattr(self, 'custom_initial_policy') and self.custom_initial_policy:
            print("Using custom initial policy")
            policy = self.custom_initial_policy.copy()
        else:
            print("Using random initial policy")
            policy = {}
            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)
                    if state not in self.terminals and state not in self.obstacles:
                        policy[state] = [np.random.choice(self.actions)]  # Initialize as list

        for iteration in range(max_iterations):
            # Policy Evaluation with step tracking
            V, eval_steps = self.policy_evaluation_with_history(policy)

            # Store policy evaluation history for this policy iteration
            self.policy_eval_history.append(eval_steps['v_history'])
            self.policy_eval_v_history.append(eval_steps['v_history'])

            # Store final state for this policy iteration
            q_vals, formulas = self.compute_q_values_and_formulas(V)
            self.v_history.append(V.copy())
            self.q_history.append(q_vals)
            self.formula_history.append(formulas)
            self.policy_history.append(policy.copy())

            # Policy Improvement
            new_policy = self.get_greedy_policy(V)

            # Check for convergence - policies are equivalent if they have the same optimal actions
            if self._policies_equivalent(new_policy, policy):
                self.converged_iteration = iteration + 1
                break

            policy = new_policy

        return self.converged_iteration or max_iterations

    def policy_evaluation_with_history(self, policy, max_iterations=100, tolerance=1e-6):
        """
        Evaluate a given policy and return both final result and step-by-step history.

        Returns:
            V: Final value function
            history: Dictionary with 'v_history' containing all intermediate steps
        """
        V = np.zeros((self.rows, self.cols))
        v_history = []

        # Initialize terminal states based on formulation
        if self.formula_type == 'BERKELEY':
            # BERKELEY: terminal states have their reward values
            for state, reward in self.terminals.items():
                V[state] = reward
        # AIMA: terminal states remain at 0 (rewards come from transitions TO them)

        # Store initial values
        v_history.append(V.copy())

        for step in range(max_iterations):
            delta = 0.0
            new_V = V.copy()

            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)

                    if state in self.terminals or state in self.obstacles:
                        continue

                    if state not in policy:
                        continue

                    # Handle policy that may contain multiple optimal actions
                    # For policy evaluation, we can use any of the optimal actions
                    # since they all have the same value by definition
                    actions = policy[state]
                    action = actions[0] if isinstance(actions, list) else actions
                    transitions = self.get_transitions(state, action)

                    if self.formula_type == 'AIMA':
                        # AIMA: Q = Σ P(s'|s,a) * (R(s,a,s') + γ*V(s'))
                        value = 0.0
                        for prob, next_state, reward in transitions:
                            value += prob * (reward + self.gamma * V[next_state])
                    else:  # BERKELEY
                        # BERKELEY: Q = R(s) + γ * Σ P(s'|s,a) * V(s')
                        step_cost_contribution = 0.0
                        expected_next_value = 0.0

                        for prob, next_state, reward in transitions:
                            # Only pay step cost if not transitioning to terminal state
                            if next_state not in self.terminals:
                                step_cost_contribution += prob * self.step_cost
                            expected_next_value += prob * V[next_state]

                        value = step_cost_contribution + self.gamma * expected_next_value

                    delta = max(delta, abs(value - V[state]))
                    new_V[state] = value

            V = new_V

            # Keep terminal states at correct values based on formulation
            if self.formula_type == 'BERKELEY':
                # BERKELEY: maintain terminal state reward values
                for state, reward in self.terminals.items():
                    V[state] = reward
            # AIMA: terminal states remain at 0

            # Store this step
            v_history.append(V.copy())

            if delta < tolerance:
                break

        return V, {'v_history': v_history}

    def _policies_equivalent(self, policy1, policy2):
        """Check if two policies are equivalent (have same optimal actions for each state)."""
        if set(policy1.keys()) != set(policy2.keys()):
            return False

        for state in policy1:
            actions1 = set(policy1[state])
            actions2 = set(policy2[state])
            if actions1 != actions2:
                return False

        return True

    def policy_evaluation(self, policy, max_iterations=100, tolerance=1e-6):
        """Evaluate a given policy (original method for compatibility)."""
        V, _ = self.policy_evaluation_with_history(policy, max_iterations, tolerance)
        return V

    def create_visualization_widgets(self):
        """Create comprehensive ipywidgets for interactive visualization with dual sliders for policy iteration."""
        # Algorithm selection
        algorithm_widget = widgets.RadioButtons(
            options=['Value Iteration', 'Policy Iteration'],
            value='Value Iteration',
            description='Algorithm:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='200px')
        )

        # Control buttons
        run_button = widgets.Button(
            description='Run Algorithm',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='120px', height='35px')
        )

        next_step_button = widgets.Button(
            description='Next Step',
            button_style='info',
            icon='step-forward',
            disabled=True,
            layout=widgets.Layout(width='100px', height='35px')
        )

        # Main iteration slider (always visible)
        iteration_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Policy Iter:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )

        # Policy evaluation slider (only for policy iteration)
        eval_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Eval Step:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )

        # Container for sliders (to show/hide eval slider)
        slider_container = widgets.VBox([iteration_slider])

        # Display options - simple checkboxes
        show_values = widgets.Checkbox(
            value=True,
            description='Values'
        )
        show_policy = widgets.Checkbox(
            value=True,
            description='Policy'
        )
        show_qvalues = widgets.Checkbox(
            value=False,
            description='Q-Values'
        )
        show_formulas = widgets.Checkbox(
            value=False,
            description='Formulas'
        )

        # Output areas
        plot_output = widgets.Output()
        text_output = widgets.Output()

        def run_algorithm(b):
            with plot_output:
                clear_output(wait=True)
                print("Running algorithm...")

            if algorithm_widget.value == 'Value Iteration':
                converged = self.value_iteration()
                # Update main slider range
                iteration_slider.max = len(self.v_history) - 1
                iteration_slider.description = 'Iteration:'
                # Hide eval slider for value iteration
                slider_container.children = [iteration_slider]
            else:
                converged = self.policy_iteration()
                # Update main slider range
                iteration_slider.max = len(self.v_history) - 1
                iteration_slider.description = 'Policy Iter:'
                # Show eval slider for policy iteration
                eval_slider.max = len(self.policy_eval_history[0]) - 1 if self.policy_eval_history else 0
                slider_container.children = [iteration_slider, eval_slider]

            iteration_slider.value = min(iteration_slider.value, iteration_slider.max)
            eval_slider.value = 0
            next_step_button.disabled = False

            with plot_output:
                clear_output(wait=True)
                print(f"Converged in {converged} iterations")
                update_display()

        def next_step(b):
            if algorithm_widget.value == 'Value Iteration':
                if iteration_slider.value < iteration_slider.max:
                    iteration_slider.value += 1
                else:
                    with plot_output:
                        clear_output(wait=True)
                        print("Already at final iteration")
                        update_display()
            else:  # Policy Iteration
                # First try to advance eval step, then policy iteration
                if eval_slider.value < eval_slider.max:
                    eval_slider.value += 1
                elif iteration_slider.value < iteration_slider.max:
                    iteration_slider.value += 1
                    # Update eval slider range for new policy iteration
                    if iteration_slider.value < len(self.policy_eval_history):
                        eval_slider.max = len(self.policy_eval_history[iteration_slider.value]) - 1
                        eval_slider.value = 0
                else:
                    with plot_output:
                        clear_output(wait=True)
                        print("Already at final iteration")
                        update_display()

        def update_display(*args):
            if not self.v_history:
                return

            policy_iter = iteration_slider.value

            if algorithm_widget.value == 'Value Iteration':
                # Use normal iteration display
                with plot_output:
                    clear_output(wait=True)
                    self.plot_iteration(
                        policy_iter,
                        show_values=show_values.value,
                        show_policy=show_policy.value,
                        show_qvalues=show_qvalues.value
                    )
                    plt.show()

                with text_output:
                    clear_output(wait=True)
                    if show_formulas.value and policy_iter < len(self.formula_history):
                        self.print_formulas(policy_iter)

            else:  # Policy Iteration
                eval_step = eval_slider.value

                with plot_output:
                    clear_output(wait=True)
                    self.plot_policy_iteration_step(
                        policy_iter, eval_step,
                        show_values=show_values.value,
                        show_policy=show_policy.value,
                        show_qvalues=show_qvalues.value
                    )
                    plt.show()

                with text_output:
                    clear_output(wait=True)
                    if show_formulas.value and policy_iter < len(self.formula_history):
                        self.print_policy_iteration_formulas(policy_iter, eval_step)

        def on_algorithm_change(change):
            """Handle algorithm selection change."""
            if change['new'] == 'Value Iteration':
                iteration_slider.description = 'Iteration:'
                slider_container.children = [iteration_slider]
            else:
                iteration_slider.description = 'Policy Iter:'
                if hasattr(self, 'policy_eval_history') and self.policy_eval_history:
                    eval_slider.max = len(self.policy_eval_history[0]) - 1
                    slider_container.children = [iteration_slider, eval_slider]
                else:
                    slider_container.children = [iteration_slider, eval_slider]
            update_display()

        def on_policy_iter_change(change):
            """Update eval slider when policy iteration changes."""
            if algorithm_widget.value == 'Policy Iteration' and hasattr(self, 'policy_eval_history'):
                policy_iter = change['new']
                if policy_iter < len(self.policy_eval_history):
                    eval_slider.max = len(self.policy_eval_history[policy_iter]) - 1
                    eval_slider.value = 0

        # Connect callbacks
        run_button.on_click(run_algorithm)
        next_step_button.on_click(next_step)
        algorithm_widget.observe(on_algorithm_change, names='value')
        iteration_slider.observe(update_display, names='value')
        iteration_slider.observe(on_policy_iter_change, names='value')
        eval_slider.observe(update_display, names='value')
        show_values.observe(update_display, names='value')
        show_policy.observe(update_display, names='value')
        show_qvalues.observe(update_display, names='value')
        show_formulas.observe(update_display, names='value')

        # Improved Layout
        # Top row: Algorithm selection and control buttons
        top_row = widgets.HBox([
            algorithm_widget,
            widgets.VBox([
                widgets.HBox([run_button, next_step_button],
                           layout=widgets.Layout(justify_content='flex-start')),
            ], layout=widgets.Layout(margin='0px 0px 0px 20px'))
        ], layout=widgets.Layout(align_items='flex-start'))

        # Middle row: Sliders (dynamic based on algorithm)
        middle_row = widgets.HBox([slider_container],
                                layout=widgets.Layout(margin='10px 0px'))

        # Bottom row: Simple checkbox row
        display_options = widgets.HBox([show_values, show_policy, show_qvalues, show_formulas])

        # Main control panel
        controls = widgets.VBox([
            top_row,
            middle_row,
            display_options
        ], layout=widgets.Layout(
            border='2px solid #4CAF50',
            padding='15px',
            margin='10px',
            border_radius='10px'
        ))

        display(controls)
        display(plot_output)
        display(text_output)

        return {
            'algorithm': algorithm_widget,
            'run_button': run_button,
            'next_step_button': next_step_button,
            'iteration_slider': iteration_slider,
            'eval_slider': eval_slider,
            'show_values': show_values,
            'show_policy': show_policy,
            'show_qvalues': show_qvalues,
            'show_formulas': show_formulas,
            'plot_output': plot_output,
            'text_output': text_output
        }

    def _get_value_color(self, value, min_val, max_val):
        """Get gradient color for a state value (green=high, yellow=low positive, red=negative, toxic green=extremely high values)."""
        # Get terminal reward range for extreme value detection
        max_terminal_reward = max(self.terminals.values()) if self.terminals else 0
        min_terminal_reward = min(self.terminals.values()) if self.terminals else 0

        # Check for extreme values first (even if all values are the same)
        if value > max_terminal_reward + 1.0:  # Extremely high values
            return '#00FF00'  # Bright toxic green
        elif value < min_terminal_reward - 1.0:  # Extremely low values
            return '#FF0000'  # Bright toxic red

        # Only return gray for uniform values that are in the "normal" range
        if max_val == min_val:
            return 'lightgray'  # All values are the same and in normal range

        # Normalize value to [0, 1] range for gradient
        normalized = (value - min_val) / (max_val - min_val)

        if value < 0:
            # Negative values: interpolate from gray to red based on how negative
            if min_val < 0:
                # Scale based on how negative compared to most negative value
                intensity = abs(value) / abs(min_val) if min_val != 0 else 0
                intensity = min(intensity, 1.0)
                return mcolors.to_hex((0.8 + 0.2*intensity, 0.8 - 0.3*intensity, 0.8 - 0.3*intensity))
            else:
                return 'lightgray'
        else:
            # Positive values: interpolate from yellow (low) to green (high)
            if max_val > 0:
                intensity = value / max_val if max_val != 0 else 0
                intensity = min(intensity, 1.0)
                # Yellow to green: (1,1,0) -> (0,1,0)
                red_component = 1.0 - 0.7*intensity    # 1.0 -> 0.3
                green_component = 1.0                  # Always 1.0
                blue_component = 0.3 - 0.3*intensity  # 0.3 -> 0.0
                return mcolors.to_hex((red_component, green_component, blue_component))
            else:
                return 'lightgray'

    def plot_iteration(self, iteration, show_values=True, show_policy=True, show_qvalues=False):
        """Plot a specific iteration with customizable display options."""
        if iteration >= len(self.v_history):
            print(f"Iteration {iteration} not available")
            return

        V = self.v_history[iteration]
        policy = self.policy_history[iteration] if iteration < len(self.policy_history) else {}
        q_values = self.q_history[iteration] if iteration < len(self.q_history) else {}

        # Increase DPI for higher resolution but reduce figure size for better fit
        fig, ax = plt.subplots(figsize=(max(4, self.cols * 0.8), max(3, self.rows * 0.8)), dpi=150)

        # Calculate value range for gradient coloring (excluding terminals and obstacles)
        regular_values = []
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state not in self.terminals and state not in self.obstacles:
                    regular_values.append(V[state])

        if regular_values:
            min_val, max_val = min(regular_values), max(regular_values)
        else:
            min_val, max_val = 0, 0

        # Draw grid
        for r in range(self.rows):
            for c in range(self.cols):
                y = self.rows - 1 - r
                x = c

                # Grid lines
                rect = patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                state = (r, c)
                center_x, center_y = x + 0.5, y + 0.5

                # Terminal states - increased font size
                if state in self.terminals:
                    reward = self.terminals[state]
                    color = 'green' if reward > 0 else 'red'
                    ax.text(center_x, center_y, f'{reward:+.0f}',
                           ha='center', va='center', fontsize=14,
                           color=color, fontweight='bold')
                    # Add background color
                    bg_color = 'lightgreen' if reward > 0 else 'lightcoral'
                    bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                              facecolor=bg_color, alpha=0.3)
                    ax.add_patch(bg_rect)

                # Obstacle states - increased font size
                elif state in self.obstacles:
                    obstacle_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                    facecolor='gray', alpha=0.7)
                    ax.add_patch(obstacle_rect)
                    ax.text(center_x, center_y, '■', ha='center', va='center',
                           fontsize=18, color='black')

                # Regular states
                else:
                    # Add gradient background color based on value
                    if show_values and regular_values:
                        bg_color = self._get_value_color(V[state], min_val, max_val)
                        bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                  facecolor=bg_color, alpha=0.4)
                        ax.add_patch(bg_rect)

                    # Show values - reduced font size and decimal places
                    if show_values:
                        ax.text(center_x, center_y, f'{V[state]:.2f}',
                               ha='center', va='center', fontsize=8, fontweight='bold')

                    # Show policy - handle multiple optimal actions - increased font size
                    if show_policy and state in policy:
                        optimal_actions = policy[state]
                        # Position policy arrows slightly below center if values are shown
                        base_policy_y = center_y - 0.2 if show_values else center_y

                        if len(optimal_actions) == 1:
                            # Single optimal action - show arrow at center
                            arrow = self.arrow_symbols[optimal_actions[0]]
                            ax.text(center_x, base_policy_y, arrow,
                                   ha='center', va='center', fontsize=14, color='blue')
                        else:
                            # Multiple optimal actions - position based on direction
                            offset = 0.08  # Small offset amount

                            for action in optimal_actions:
                                arrow = self.arrow_symbols[action]

                                # Position arrows based on their direction
                                if action == 'U':
                                    arrow_x, arrow_y = center_x, base_policy_y + offset
                                elif action == 'D':
                                    arrow_x, arrow_y = center_x, base_policy_y - offset
                                elif action == 'L':
                                    arrow_x, arrow_y = center_x - offset, base_policy_y
                                elif action == 'R':
                                    arrow_x, arrow_y = center_x + offset, base_policy_y
                                else:
                                    arrow_x, arrow_y = center_x, base_policy_y

                                ax.text(arrow_x, arrow_y, arrow,
                                       ha='center', va='center', fontsize=14,
                                       color='blue')

                    # Show Q-values as small triangular indicators at cell edges - increased font size
                    if show_qvalues and state in q_values:
                        # Define positions for triangular Q-value indicators
                        triangle_size = 0.08  # Size of triangular indicators

                        # Triangle positions at edges
                        triangle_positions = {
                            'U': (center_x, y + 0.95),      # Top edge
                            'D': (center_x, y + 0.05),      # Bottom edge
                            'L': (x + 0.05, center_y),      # Left edge
                            'R': (x + 0.95, center_y)       # Right edge
                        }

                        # Text positions closer to center
                        text_positions = {
                            'U': (center_x, y + 0.82),      # Closer to center from top
                            'D': (center_x, y + 0.18),      # Closer to center from bottom
                            'L': (x + 0.18, center_y),      # Closer to center from left
                            'R': (x + 0.82, center_y)       # Closer to center from right
                        }

                        # Triangle orientations (pointing towards the action direction)
                        triangle_orientations = {
                            'U': [(-triangle_size, -triangle_size*0.8), (triangle_size, -triangle_size*0.8), (0, triangle_size*0.8)],
                            'D': [(-triangle_size, triangle_size*0.8), (triangle_size, triangle_size*0.8), (0, -triangle_size*0.8)],
                            'L': [(triangle_size*0.8, -triangle_size), (triangle_size*0.8, triangle_size), (-triangle_size*0.8, 0)],
                            'R': [(-triangle_size*0.8, -triangle_size), (-triangle_size*0.8, triangle_size), (triangle_size*0.8, 0)]
                        }

                        for action in self.actions:
                            if action in q_values[state]:
                                triangle_x, triangle_y = triangle_positions[action]
                                text_x, text_y = text_positions[action]
                                q_val = q_values[state][action]

                                # Draw small triangle
                                triangle_points = triangle_orientations[action]
                                triangle_coords = [(triangle_x + dx, triangle_y + dy) for dx, dy in triangle_points]
                                triangle = patches.Polygon(triangle_coords, closed=True,
                                                         facecolor='lightblue', edgecolor='darkblue',
                                                         alpha=0.7, linewidth=0.5)
                                ax.add_patch(triangle)

                                # Add Q-value text closer to center - significantly increased font size
                                ax.text(text_x, text_y, f'{q_val:.2f}',
                                       ha='center', va='center', fontsize=6,
                                       color='darkblue', fontweight='bold')

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title with step cost info - positioned at left edge of plot
        title_parts = [f'Iteration {iteration}']
        if self.step_cost != 0:
            title_parts.append(f'(Step Cost: {self.step_cost:.2f})')
        if show_values:
            title_parts.append('Values')
        if show_policy:
            title_parts.append('Policy')
        if show_qvalues:
            title_parts.append('Q-Values')

        # Position title at left edge of plot area
        ax.text(0, self.rows + 0.15, ' + '.join(title_parts),
                fontsize=12, fontweight='bold', ha='left', va='bottom',
                transform=ax.transData)
        ax.text(self.cols/2, -0.15, f'γ={self.gamma}, noise={self.noise}, step_cost={self.step_cost:.2f}',
                ha='center', va='top', fontsize=10, transform=ax.transData)

    def plot_policy_iteration_step(self, policy_iter, eval_step, show_values=True, show_policy=True, show_qvalues=False):
        """Plot a specific policy iteration step showing intermediate policy evaluation."""
        if not hasattr(self, 'policy_eval_history') or policy_iter >= len(self.policy_eval_history):
            print(f"Policy iteration {policy_iter} not available")
            return

        if eval_step >= len(self.policy_eval_history[policy_iter]):
            print(f"Evaluation step {eval_step} not available for policy iteration {policy_iter}")
            return

        # Get values from policy evaluation step
        V = self.policy_eval_history[policy_iter][eval_step]

        # Get policy for this policy iteration (fixed during evaluation)
        policy = self.policy_history[policy_iter] if policy_iter < len(self.policy_history) else {}

        # Compute Q-values for current values and policy
        q_values = {}
        if show_qvalues:
            q_values, _ = self.compute_q_values_and_formulas(V)

        # Increase DPI for higher resolution but reduce figure size for better fit
        fig, ax = plt.subplots(figsize=(max(4, self.cols * 0.8), max(3, self.rows * 0.8)), dpi=150)

        # Calculate value range for gradient coloring (excluding terminals and obstacles)
        regular_values = []
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state not in self.terminals and state not in self.obstacles:
                    regular_values.append(V[state])

        if regular_values:
            min_val, max_val = min(regular_values), max(regular_values)
        else:
            min_val, max_val = 0, 0

        # Draw grid (similar to plot_iteration but with policy iteration specific styling)
        for r in range(self.rows):
            for c in range(self.cols):
                y = self.rows - 1 - r
                x = c

                # Grid lines
                rect = patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                state = (r, c)
                center_x, center_y = x + 0.5, y + 0.5

                # Terminal states
                if state in self.terminals:
                    reward = self.terminals[state]
                    color = 'green' if reward > 0 else 'red'
                    ax.text(center_x, center_y, f'{reward:+.0f}',
                           ha='center', va='center', fontsize=16,
                           color=color, fontweight='bold')
                    bg_color = 'lightgreen' if reward > 0 else 'lightcoral'
                    bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                              facecolor=bg_color, alpha=0.3)
                    ax.add_patch(bg_rect)

                # Obstacle states
                elif state in self.obstacles:
                    obstacle_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                    facecolor='gray', alpha=0.7)
                    ax.add_patch(obstacle_rect)
                    ax.text(center_x, center_y, '■', ha='center', va='center',
                           fontsize=16, color='black')

                # Regular states
                else:
                    # Add gradient background color based on value
                    if show_values and regular_values:
                        bg_color = self._get_value_color(V[state], min_val, max_val)
                        bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                  facecolor=bg_color, alpha=0.4)
                        ax.add_patch(bg_rect)

                    # Show values - reduced font size and decimal places
                    if show_values:
                        ax.text(center_x, center_y, f'{V[state]:.2f}',
                               ha='center', va='center', fontsize=8, fontweight='bold')

                    # Show policy (fixed during policy evaluation)
                    if show_policy and state in policy:
                        optimal_actions = policy[state]
                        base_policy_y = center_y - 0.2 if show_values else center_y

                        if len(optimal_actions) == 1:
                            arrow = self.arrow_symbols[optimal_actions[0]]
                            ax.text(center_x, base_policy_y, arrow,
                                   ha='center', va='center', fontsize=16, color='purple')  # Purple for fixed policy
                        else:
                            offset = 0.08
                            for action in optimal_actions:
                                arrow = self.arrow_symbols[action]
                                if action == 'U':
                                    arrow_x, arrow_y = center_x, base_policy_y + offset
                                elif action == 'D':
                                    arrow_x, arrow_y = center_x, base_policy_y - offset
                                elif action == 'L':
                                    arrow_x, arrow_y = center_x - offset, base_policy_y
                                elif action == 'R':
                                    arrow_x, arrow_y = center_x + offset, base_policy_y
                                else:
                                    arrow_x, arrow_y = center_x, base_policy_y

                                ax.text(arrow_x, arrow_y, arrow,
                                       ha='center', va='center', fontsize=16, color='purple')

                    # Show Q-values (same as regular plot_iteration)
                    if show_qvalues and state in q_values:
                        triangle_size = 0.08
                        triangle_positions = {
                            'U': (center_x, y + 0.95), 'D': (center_x, y + 0.05),
                            'L': (x + 0.05, center_y), 'R': (x + 0.95, center_y)
                        }
                        text_positions = {
                            'U': (center_x, y + 0.82), 'D': (center_x, y + 0.18),
                            'L': (x + 0.18, center_y), 'R': (x + 0.82, center_y)
                        }
                        triangle_orientations = {
                            'U': [(-triangle_size, -triangle_size*0.8), (triangle_size, -triangle_size*0.8), (0, triangle_size*0.8)],
                            'D': [(-triangle_size, triangle_size*0.8), (triangle_size, triangle_size*0.8), (0, -triangle_size*0.8)],
                            'L': [(triangle_size*0.8, -triangle_size), (triangle_size*0.8, triangle_size), (-triangle_size*0.8, 0)],
                            'R': [(-triangle_size*0.8, -triangle_size), (-triangle_size*0.8, triangle_size), (triangle_size*0.8, 0)]
                        }

                        for action in self.actions:
                            if action in q_values[state]:
                                triangle_x, triangle_y = triangle_positions[action]
                                text_x, text_y = text_positions[action]
                                q_val = q_values[state][action]

                                triangle_points = triangle_orientations[action]
                                triangle_coords = [(triangle_x + dx, triangle_y + dy) for dx, dy in triangle_points]
                                triangle = patches.Polygon(triangle_coords, closed=True,
                                                         facecolor='lightblue', edgecolor='darkblue',
                                                         alpha=0.7, linewidth=0.5)
                                ax.add_patch(triangle)

                                ax.text(text_x, text_y, f'{q_val:.2f}',
                                       ha='center', va='center', fontsize=6,
                                       color='darkblue', fontweight='bold')

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title with policy iteration info
        is_final_eval_step = eval_step == len(self.policy_eval_history[policy_iter]) - 1
        eval_status = "CONVERGED" if is_final_eval_step else f"Step {eval_step}"

        title_parts = [f'Policy Iter {policy_iter} | Policy Eval {eval_status}']
        if self.step_cost != 0:
            title_parts.append(f'(Step Cost: {self.step_cost:.2f})')
        if show_values:
            title_parts.append('Values')
        if show_policy:
            title_parts.append('Policy (Fixed)')
        if show_qvalues:
            title_parts.append('Q-Values')

        # Position title at left edge of plot area
        ax.text(0, self.rows + 0.15, ' + '.join(title_parts),
                fontsize=12, fontweight='bold', ha='left', va='bottom',
                transform=ax.transData)
        ax.text(self.cols/2, -0.15, f'γ={self.gamma}, noise={self.noise}, step_cost={self.step_cost:.2f}',
                ha='center', va='top', fontsize=12, transform=ax.transData)

    def print_policy_iteration_formulas(self, policy_iter, eval_step):
        """Print formulas for a specific policy iteration and evaluation step."""
        if not hasattr(self, 'policy_eval_history') or policy_iter >= len(self.policy_eval_history):
            print(f"Policy iteration {policy_iter} not available")
            return

        if eval_step >= len(self.policy_eval_history[policy_iter]):
            print(f"Evaluation step {eval_step} not available for policy iteration {policy_iter}")
            return

        V = self.policy_eval_history[policy_iter][eval_step]
        policy = self.policy_history[policy_iter] if policy_iter < len(self.policy_history) else {}

        # Compute formulas for current state
        _, formulas = self.compute_q_values_and_formulas(V)

        is_final_eval_step = eval_step == len(self.policy_eval_history[policy_iter]) - 1
        eval_status = "CONVERGED" if is_final_eval_step else f"Step {eval_step}"

        print(f"\n{'='*60}")
        print(f"POLICY ITERATION FORMULAS")
        print(f"Policy Iteration {policy_iter} | Policy Evaluation {eval_status}")
        if self.step_cost != 0:
            print(f"Step Cost: {self.step_cost:+.1f}")
        print(f"{'='*60}")

        print(f"\nCurrent Policy (Fixed during evaluation):")
        for state, actions in policy.items():
            if len(actions) == 1:
                print(f"  π({state}) = {actions[0]}")
            else:
                print(f"  π({state}) = {actions} (tied optimal)")

        print(f"\nQ-value Formulas:")
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state in formulas:
                    print(f"\nState {state}:")
                    for action in self.actions:
                        if action in formulas[state]:
                            # Highlight the action(s) chosen by current policy
                            is_policy_action = action in policy.get(state, [])
                            prefix = ">>> " if is_policy_action else "    "
                            print(f"{prefix}Q({state},{action}) = {formulas[state][action]}")

        # Show value function convergence info
        if eval_step > 0:
            prev_V = self.policy_eval_history[policy_iter][eval_step - 1]
            max_change = 0.0
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) not in self.terminals and (r, c) not in self.obstacles:
                        change = abs(V[r, c] - prev_V[r, c])
                        max_change = max(max_change, change)
            print(f"\nMax Value Change from Previous Step: {max_change:.6f}")

        if is_final_eval_step:
            print(f"\n>>> POLICY EVALUATION CONVERGED <<<")

    def print_formulas(self, iteration):
        """Print Q-value formulas for a specific iteration."""
        if iteration >= len(self.formula_history):
            print(f"Formulas for iteration {iteration} not available")
            return

        formulas = self.formula_history[iteration]

        print(f"\n{'='*50}")
        print(f"Q-VALUE FORMULAS - ITERATION {iteration}")
        if self.step_cost != 0:
            print(f"Step Cost: {self.step_cost:+.1f}")
        print(f"{'='*50}")

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)

                if state in formulas:
                    print(f"\nState {state}:")
                    for action in self.actions:
                        if action in formulas[state]:
                            print(f"  Q({state},{action}) = {formulas[state][action]}")

    def plot_environment(self):
        """Plot just the basic environment structure - terminals, obstacles, and empty states."""
        # Increase DPI for higher resolution but reduce figure size for better fit
        fig, ax = plt.subplots(figsize=(max(4, self.cols * 0.8), max(3, self.rows * 0.8)), dpi=150)

        # Draw grid
        for r in range(self.rows):
            for c in range(self.cols):
                y = self.rows - 1 - r
                x = c

                # Grid lines
                rect = patches.Rectangle((x, y), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)

                state = (r, c)
                center_x, center_y = x + 0.5, y + 0.5

                # Terminal states - increased font size
                if state in self.terminals:
                    reward = self.terminals[state]
                    color = 'green' if reward > 0 else 'red'
                    ax.text(center_x, center_y, f'{reward:+.0f}',
                           ha='center', va='center', fontsize=16,
                           color=color, fontweight='bold')
                    # Add background color
                    bg_color = 'lightgreen' if reward > 0 else 'lightcoral'
                    bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                              facecolor=bg_color, alpha=0.3)
                    ax.add_patch(bg_rect)

                # Obstacle states - increased font size
                elif state in self.obstacles:
                    obstacle_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                    facecolor='gray', alpha=0.7)
                    ax.add_patch(obstacle_rect)
                    ax.text(center_x, center_y, '■', ha='center', va='center',
                           fontsize=16, color='black')

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks([])
        ax.set_yticks([])

        # Position title at left edge of plot area
        ax.text(0, self.rows + 0.15, f'Grid World Environment ({self.rows}x{self.cols})',
                fontsize=14, fontweight='bold', ha='left', va='bottom',
                transform=ax.transData)
        ax.text(self.cols/2, -0.15, f'γ={self.gamma}, noise={self.noise}, step_cost={self.step_cost:.2f}',
                ha='center', va='top', fontsize=12, transform=ax.transData)
        plt.show()


def create_complex_maze_environment(
    noise=0.2,
    step_cost=1,
    gamma=0.95

):
    """
    Create a complex maze environment to use for testing the MDP solver.
    """

    rows = 20
    cols = 23

    obstacles = set()

    # Top border walls
    for c in range(cols):
        obstacles.add((0, c))

    # Bottom border walls
    for c in range(cols):
        obstacles.add((rows-1, c))

    # Left border walls
    for r in range(rows):
        obstacles.add((r, 0))

    # Right border walls
    for r in range(rows):
        obstacles.add((r, cols-1))

    # Row #2
    for c in range(cols):
        obstacles.add((2, c))
    obstacles.remove((2, 1))
    obstacles.remove((2, 16))

    # Row #4
    for c in range(cols):
        obstacles.add((4, c))
    obstacles.remove((4, 1))
    obstacles.remove((4, 5))

    # Column #2
    for r in range(rows):
        obstacles.add((r, 2))
    obstacles.remove((1, 2))
    obstacles.remove((18, 2))

    # Row #17
    for c in range(cols):
        obstacles.add((17, c))
    obstacles.remove((17, 1))
    obstacles.remove((17, 21))

    # Row #5
    obstacles.add((5, 10))

    # Row #6
    obstacles.add((6, 5))
    obstacles.add((6, 6))
    obstacles.add((6, 7))
    obstacles.add((6, 17))
    obstacles.add((6, 18))
    obstacles.add((6, 19))

    # Row #7
    for c in range(10, 18):
        obstacles.add((7, c))

    # Row #8
    obstacles.add((8, 10))
    obstacles.add((8, 16))
    obstacles.add((8, 17))

    # Row #9
    obstacles.add((9, 10))
    obstacles.add((9, 16))
    obstacles.add((9, 17))
    obstacles.add((9, 18))
    obstacles.add((9, 20))
    obstacles.add((9, 21))

    # Row #10
    obstacles.add((10, 6))
    obstacles.add((10, 7))
    obstacles.add((10, 8))
    obstacles.add((10, 9))
    obstacles.add((10, 10))
    obstacles.add((10, 16))

    # Row #11
    obstacles.add((11, 4))
    obstacles.add((11, 12))
    obstacles.add((11, 16))

    # Row #12
    obstacles.add((12, 4))
    obstacles.add((12, 12))
    obstacles.add((12, 14))
    obstacles.add((12, 16))
    obstacles.add((12, 17))
    obstacles.add((12, 18))
    obstacles.add((12, 19))
    obstacles.add((12, 20))

    # Row #13
    obstacles.add((13, 4))
    obstacles.add((13, 5))
    obstacles.add((13, 7))
    obstacles.add((13, 12))
    obstacles.add((13, 16))

    # Row #14
    obstacles.add((14, 7))
    obstacles.add((14, 12))
    obstacles.add((14, 16))

    # Row #15
    obstacles.add((15, 3))
    obstacles.add((15, 4))
    obstacles.add((15, 5))
    obstacles.add((15, 7))
    obstacles.add((15, 8))
    obstacles.add((15, 9))
    obstacles.add((15, 10))
    obstacles.add((15, 11))
    obstacles.add((15, 12))
    obstacles.add((15, 19))
    obstacles.add((15, 20))
    obstacles.add((15, 21))

    # Row #16
    obstacles.add((16, 11))

    terminals = {
        (15, 17): +1000,

        (6, 16): -5,
        (8, 21): -5,
        (10, 17): -8,
        (13, 17): -12,

        (9, 6): -6,
        (7, 8): -20,
        (8, 8): -10,

        (11, 10): -10,
        (12, 8): -20,
        (13, 11): -5,
        (14, 10): -20,

        (10, 12): -10,
        (10, 13): -20,
        (10, 14): -15,
        (9, 14): -20,
        (12, 15): -20,
        (14, 13): -10,
        (15, 15): -8
    }

    # Create the environment
    env = GridWorldMDP(rows=rows, cols=cols, gamma=gamma,
                       noise=noise, step_cost=step_cost)
    env.set_terminals(terminals)
    env.set_obstacles(obstacles)

    return env
