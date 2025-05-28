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

    def __init__(self, rows=3, cols=4, gamma=0.9, noise=0.1, step_cost=0.0, formula_type='BERKELEY'):
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

    def reset_history(self):
        """Reset all iteration history."""
        self.v_history = []
        self.q_history = []
        self.formula_history = []
        self.policy_history = []
        self.converged_iteration = None

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
        intended_prob = 1.0 - 2 * self.noise
        side_prob = self.noise

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

    def get_greedy_policy(self, V):
        """Extract greedy policy from value function."""
        policy = {}

        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)

                if state in self.terminals or state in self.obstacles:
                    continue

                best_action = None
                best_value = -np.inf

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

                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                policy[state] = best_action

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

        Returns:
            converged_iteration: The iteration where convergence occurred
        """
        self.reset_history()

        # Initialize random policy
        policy = {}
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state not in self.terminals and state not in self.obstacles:
                    policy[state] = np.random.choice(self.actions)

        for iteration in range(max_iterations):
            # Policy Evaluation
            V = self.policy_evaluation(policy)

            # Store current state
            q_vals, formulas = self.compute_q_values_and_formulas(V)
            self.v_history.append(V.copy())
            self.q_history.append(q_vals)
            self.formula_history.append(formulas)
            self.policy_history.append(policy.copy())

            # Policy Improvement
            new_policy = self.get_greedy_policy(V)

            # Check for convergence
            if new_policy == policy:
                self.converged_iteration = iteration + 1
                break

            policy = new_policy

        return self.converged_iteration or max_iterations

    def policy_evaluation(self, policy, max_iterations=100, tolerance=1e-6):
        """Evaluate a given policy."""
        V = np.zeros((self.rows, self.cols))

        # Initialize terminal states based on formulation
        if self.formula_type == 'BERKELEY':
            # BERKELEY: terminal states have their reward values
            for state, reward in self.terminals.items():
                V[state] = reward
        # AIMA: terminal states remain at 0 (rewards come from transitions TO them)

        for _ in range(max_iterations):
            delta = 0.0
            new_V = V.copy()

            for r in range(self.rows):
                for c in range(self.cols):
                    state = (r, c)

                    if state in self.terminals or state in self.obstacles:
                        continue

                    if state not in policy:
                        continue

                    action = policy[state]
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

            if delta < tolerance:
                break

        return V

    def create_visualization_widgets(self):
        """Create comprehensive ipywidgets for interactive visualization."""
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

        # Iteration slider (will be updated after running)
        iteration_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description='Iteration:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='400px')
        )

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
            else:
                converged = self.policy_iteration()

            # Update slider range and enable next step button
            iteration_slider.max = len(self.v_history) - 1
            iteration_slider.value = min(iteration_slider.value, iteration_slider.max)
            next_step_button.disabled = False

            with plot_output:
                clear_output(wait=True)
                print(f"Converged in {converged} iterations")
                update_display()

        def next_step(b):
            if iteration_slider.value < iteration_slider.max:
                iteration_slider.value += 1
            else:
                with plot_output:
                    clear_output(wait=True)
                    print("Already at final iteration")
                    update_display()

        def update_display(*args):
            if not self.v_history:
                return

            iteration = iteration_slider.value

            with plot_output:
                clear_output(wait=True)
                self.plot_iteration(
                    iteration,
                    show_values=show_values.value,
                    show_policy=show_policy.value,
                    show_qvalues=show_qvalues.value
                )
                plt.show()

            with text_output:
                clear_output(wait=True)
                if show_formulas.value and iteration < len(self.formula_history):
                    self.print_formulas(iteration)

        # Connect callbacks
        run_button.on_click(run_algorithm)
        next_step_button.on_click(next_step)
        iteration_slider.observe(update_display, names='value')
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

        # Middle row: Iteration slider
        middle_row = widgets.HBox([iteration_slider],
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

        fig, ax = plt.subplots(figsize=(max(6, self.cols * 1.5), max(4, self.rows * 1.2)))

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

                # Terminal states
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

                # Obstacle states
                elif state in self.obstacles:
                    obstacle_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                    facecolor='gray', alpha=0.7)
                    ax.add_patch(obstacle_rect)
                    ax.text(center_x, center_y, '■', ha='center', va='center',
                           fontsize=20, color='black')

                # Regular states
                else:
                    # Add gradient background color based on value
                    if show_values and regular_values:
                        bg_color = self._get_value_color(V[state], min_val, max_val)
                        bg_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                  facecolor=bg_color, alpha=0.4)
                        ax.add_patch(bg_rect)

                    # Show values
                    if show_values:
                        ax.text(center_x, center_y, f'{V[state]:.3f}',
                               ha='center', va='center', fontsize=10, fontweight='bold')

                    # Show policy
                    if show_policy and state in policy:
                        arrow = self.arrow_symbols[policy[state]]
                        # Position policy arrow slightly below center if values are shown
                        policy_y = center_y - 0.2 if show_values else center_y
                        ax.text(center_x, policy_y, arrow,
                               ha='center', va='center', fontsize=14, color='blue')

                    # Show Q-values as small triangular indicators at cell edges
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

                                # Add Q-value text closer to center
                                ax.text(text_x, text_y, f'{q_val:.2f}',
                                       ha='center', va='center', fontsize=5,
                                       color='darkblue', fontweight='bold')

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title with step cost info
        title_parts = [f'Iteration {iteration}']
        if self.step_cost != 0:
            title_parts.append(f'(Step Cost: {self.step_cost:.2f})')
        if show_values:
            title_parts.append('Values')
        if show_policy:
            title_parts.append('Policy')
        if show_qvalues:
            title_parts.append('Q-Values')

        ax.set_title(' + '.join(title_parts), fontsize=14, fontweight='bold')
        ax.text(self.cols/2, -0.15, f'γ={self.gamma}, noise={self.noise}, step_cost={self.step_cost:.2f}',
                ha='center', va='top', fontsize=10, transform=ax.transData)

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
        fig, ax = plt.subplots(figsize=(max(6, self.cols * 1.5), max(4, self.rows * 1.2)))

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

                # Terminal states
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

                # Obstacle states
                elif state in self.obstacles:
                    obstacle_rect = patches.Rectangle((x, y), 1, 1, fill=True,
                                                    facecolor='gray', alpha=0.7)
                    ax.add_patch(obstacle_rect)
                    ax.text(center_x, center_y, '■', ha='center', va='center',
                           fontsize=20, color='black')

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f'Grid World Environment ({self.rows}x{self.cols})', fontsize=14, fontweight='bold')
        ax.text(self.cols/2, -0.15, f'γ={self.gamma}, noise={self.noise}, step_cost={self.step_cost:.2f}',
                ha='center', va='top', fontsize=10, transform=ax.transData)
        plt.show()


def create_classic_gridworld_env(formula_type='BERKELEY'):
    """Create a demo environment with default settings."""
    env = GridWorldMDP(rows=3, cols=4, gamma=0.9, noise=0.1, step_cost=0.0, formula_type=formula_type)

    # Set up classic 3x4 grid world
    env.set_terminals({(0, 3): +1, (1, 3): -1})
    env.set_obstacles({(1, 1)})  # Classic example has one obstacle

    return env


def create_custom_environment(rows, cols, terminals, obstacles=None, gamma=0.9, noise=0.1, step_cost=0.0, formula_type='BERKELEY'):
    """
    Create a custom grid world environment.

    Args:
        rows (int): Number of rows
        cols (int): Number of columns
        terminals (dict): Dictionary of {(row, col): reward}
        obstacles (set): Set of obstacle coordinates (row, col), or None for no obstacles
        gamma (float): Discount factor
        noise (float): Action noise probability
        step_cost (float): Cost/reward for each step (negative = cost, positive = reward)
        formula_type (str): 'AIMA' or 'BERKELEY' - determines reward calculation method

    Returns:
        GridWorldMDP: Configured environment
    """
    env = GridWorldMDP(rows=rows, cols=cols, gamma=gamma, noise=noise, step_cost=step_cost, formula_type=formula_type)
    env.set_terminals(terminals)
    env.set_obstacles(obstacles)
    return env


# Simple debugging helper functions
def debug_value_iteration(formula_type='BERKELEY'):
    """Simple function to debug value iteration step by step."""
    env = create_classic_gridworld_env(formula_type=formula_type)

    print("=== Value Iteration Debug ===")
    print(f"Environment: {env.rows}x{env.cols}, gamma={env.gamma}")
    print(f"Formula type: {env.formula_type}")

    # Set breakpoint on next line and step into value_iteration method
    iterations = env.value_iteration(max_iterations=5, tolerance=1e-6)

    print(f"Converged in {iterations} iterations")
    print("Final values:")
    print(env.v_history[-1])

    return env


def debug_policy_iteration(formula_type='BERKELEY'):
    """Simple function to debug policy iteration step by step."""
    env = create_classic_gridworld_env(formula_type=formula_type)

    print("=== Policy Iteration Debug ===")
    print(f"Environment: {env.rows}x{env.cols}, gamma={env.gamma}")
    print(f"Formula type: {env.formula_type}")

    # Set breakpoint on next line and step into policy_iteration method
    iterations = env.policy_iteration(max_iterations=5)

    print(f"Converged in {iterations} iterations")
    print("Final values:")
    print(env.v_history[-1])

    return env


def show_iteration_details(env, iteration_num):
    """Show details for a specific iteration after running an algorithm."""
    if iteration_num >= len(env.v_history):
        print(f"Iteration {iteration_num} not available")
        return

    print(f"\n--- Iteration {iteration_num} ({env.formula_type} formula) ---")
    print("Values:")
    print(env.v_history[iteration_num])

    if iteration_num < len(env.policy_history):
        print("Policy:")
        for state, action in env.policy_history[iteration_num].items():
            print(f"  {state}: {action}")


def compare_formulations():
    """Compare AIMA and BERKELEY formulations side by side."""
    print("=== COMPARING AIMA vs BERKELEY FORMULATIONS ===")

    # AIMA formulation
    env_aima = create_classic_gridworld_env()
    env_aima.formula_type = 'AIMA'
    iterations_aima = env_aima.value_iteration(max_iterations=10)

    # BERKELEY formulation
    env_berkeley = create_classic_gridworld_env()
    env_berkeley.formula_type = 'BERKELEY'
    iterations_berkeley = env_berkeley.value_iteration(max_iterations=10)

    print(f"\nAIMA converged in {iterations_aima} iterations")
    print(f"BERKELEY converged in {iterations_berkeley} iterations")

    print(f"\nFinal values - AIMA:")
    print(env_aima.v_history[-1])

    print(f"\nFinal values - BERKELEY:")
    print(env_berkeley.v_history[-1])

    # Check if results are similar
    import numpy as np
    values_similar = np.allclose(env_aima.v_history[-1], env_berkeley.v_history[-1], atol=1e-3)
    policies_same = env_aima.policy_history[-1] == env_berkeley.policy_history[-1]

    print(f"\nValues are similar: {values_similar}")
    print(f"Policies are identical: {policies_same}")

    return env_aima, env_berkeley


# Demo usage function
def demo(formula_type='BERKELEY'):
    """Run a demonstration of the grid world MDP."""
    print("Creating demo Grid World MDP...")
    env = create_classic_gridworld_env(formula_type=formula_type)

    print("Grid Configuration:")
    print(f"  Size: {env.rows}x{env.cols}")
    print(f"  Terminals: {env.terminals}")
    print(f"  Obstacles: {env.obstacles}")
    print(f"  Gamma: {env.gamma}")
    print(f"  Noise: {env.noise}")
    print(f"  Step Cost: {env.step_cost}")

    # Create and display widgets
    widgets = env.create_visualization_widgets()

    return env, widgets


if __name__ == "__main__":

    # Debug value iteration
    env = debug_value_iteration(formula_type='AIMA')
    # Look at specific iterations
    show_iteration_details(env, 0)
    show_iteration_details(env, 1)
    show_iteration_details(env, 2)

    # Debug value iteration
    env = debug_value_iteration(formula_type='BERKELEY')
    # Look at specific iterations
    show_iteration_details(env, 0)
    show_iteration_details(env, 1)
    show_iteration_details(env, 2)

    env = debug_policy_iteration(formula_type='BERKELEY')
    # Look at specific iterations
    show_iteration_details(env, 0)
    show_iteration_details(env, 1)
    show_iteration_details(env, 2)

    # Compare formulations
    compare_formulations()
