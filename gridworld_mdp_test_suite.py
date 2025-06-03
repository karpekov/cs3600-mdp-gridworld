"""
This file is used to test the gridworld_mdp.py file.

It is used to test the following features:
- Value iteration
- Policy iteration
- Custom initial policies
- Policy convergence
- Tie-breaking
"""

import numpy as np
import matplotlib.pyplot as plt
from gridworld_mdp import GridWorldMDP


def create_classic_gridworld_env(formula_type='BERKELEY'):
    """Create a demo environment with default settings."""
    env = GridWorldMDP(rows=3, cols=4, gamma=0.9, noise=0.2,
                       step_cost=0.0, formula_type=formula_type)

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
    env = GridWorldMDP(rows=rows, cols=cols, gamma=gamma, noise=noise,
                       step_cost=step_cost, formula_type=formula_type)
    env.set_terminals(terminals)
    env.set_obstacles(obstacles)
    return env


# Simple debugging helper functions
def debug_value_iteration(formula_type='AIMA'):
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


def debug_policy_iteration(formula_type='AIMA'):
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
        for state, actions in env.policy_history[iteration_num].items():
            if len(actions) == 1:
                print(f"  {state}: {actions[0]}")
            else:
                print(f"  {state}: {actions} (tied optimal actions)")


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
    values_similar = np.allclose(
        env_aima.v_history[-1], env_berkeley.v_history[-1], atol=1e-3)
    policies_same = env_aima._policies_equivalent(
        env_aima.policy_history[-1], env_berkeley.policy_history[-1])

    print(f"\nValues are similar: {values_similar}")
    print(f"Policies are identical: {policies_same}")

    return env_aima, env_berkeley


def test_extreme_ties():
    """Test with extreme ties - environment where all actions are equally good."""
    print("\n=== Testing Extreme Ties (All Actions Equal) ===")

    # Create a 2x2 grid with high step cost and distant terminal to make all actions equal
    env = GridWorldMDP(rows=2, cols=2, gamma=0.5, noise=0.0,
                       step_cost=-0.1, formula_type='BERKELEY')
    env.set_terminals({(1, 1): +1})  # Terminal at bottom right
    env.set_obstacles(set())  # No obstacles

    # Manually set initial values to create ties
    print("Environment setup:")
    print(f"  Size: {env.rows}x{env.cols}")
    print(f"  Terminals: {env.terminals}")
    print(f"  Low gamma and step cost to create ties")

    # Run just a couple iterations to see ties
    iterations = env.value_iteration(max_iterations=3)
    print(f"Ran {iterations} iterations")

    # Show final policies
    print("\nPolicies by iteration:")
    for i, policy in enumerate(env.policy_history):
        print(f"Iteration {i}:")
        for state, actions in policy.items():
            if len(actions) == 1:
                print(f"  {state}: {actions[0]}")
            else:
                print(f"  {state}: {actions} (tied optimal actions)")

    # Plot the final iteration
    env.plot_iteration(-1, show_values=True,
                       show_policy=True, show_qvalues=False)
    plt.title("Extreme Ties - Multiple Overlapping Arrows",
              fontsize=12, ha='left')
    plt.savefig('extreme_ties_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nExtreme ties visualization saved to: extreme_ties_visualization.png")
    plt.close()

    return env


def test_ties():
    """Test function to demonstrate tie-breaking with multiple optimal actions."""
    print("=== Testing Tie-Breaking Functionality ===")

    # Create a simple 3x3 grid with terminal in the center to create more ties
    env = GridWorldMDP(rows=3, cols=3, gamma=0.9, noise=0.0,
                       step_cost=0.0, formula_type='BERKELEY')
    env.set_terminals({(1, 1): +1})  # Terminal in center
    env.set_obstacles(set())  # No obstacles

    print("Environment setup:")
    print(f"  Size: {env.rows}x{env.cols}")
    print(f"  Terminals: {env.terminals}")
    print(f"  No noise, no step cost")

    # Run value iteration
    iterations = env.value_iteration(max_iterations=10)
    print(f"Converged in {iterations} iterations")

    # Show final values and policies
    print("\nFinal values:")
    final_values = env.v_history[-1]
    for r in range(env.rows):
        for c in range(env.cols):
            print(f"  ({r},{c}): {final_values[r, c]:.3f}")

    print("\nFinal policy:")
    final_policy = env.policy_history[-1]
    for state, actions in final_policy.items():
        if len(actions) == 1:
            print(f"  {state}: {actions[0]}")
        else:
            print(f"  {state}: {actions} (tied optimal actions)")

    # Plot the environment and save to file
    env.plot_iteration(-1, show_values=True,
                       show_policy=True, show_qvalues=False)
    plt.title("Test Environment - Overlapping Arrows for Tied Optimal Actions",
              fontsize=12, ha='left')
    plt.savefig('tie_breaking_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: tie_breaking_visualization.png")
    plt.close()  # Close the figure to free memory

    return env


def test_new_policy_iteration_visualization():
    """Test the new dual-slider policy iteration visualization."""
    print("=== Testing New Policy Iteration Visualization ===")

    # Create a simple environment
    env = create_classic_gridworld_env(formula_type='BERKELEY')

    # Run policy iteration to populate history
    iterations = env.policy_iteration(max_iterations=3)
    print(f"Policy iteration converged in {iterations} iterations")

    # Test that we have policy evaluation history
    print(f"Policy eval history length: {len(env.policy_eval_history)}")
    for i, eval_steps in enumerate(env.policy_eval_history):
        print(f"  Policy iter {i}: {len(eval_steps)} evaluation steps")

    # Test plotting a specific step
    if env.policy_eval_history:
        print("\nTesting plot_policy_iteration_step...")
        env.plot_policy_iteration_step(
            0, 0, show_values=True, show_policy=True)
        plt.title("Test: Policy Iteration Step Visualization",
                  fontsize=12, ha='left')
        plt.savefig('test_policy_iteration_viz.png',
                    dpi=150, bbox_inches='tight')
        print("Test visualization saved to: test_policy_iteration_viz.png")
        plt.close()

    print("Test completed successfully!")
    return env


def demo_policy_convergence(formula_type='BERKELEY'):
    """
    Demonstrate that policy iteration converges to optimal policy from any starting policy.
    """
    print("="*70)
    print("POLICY ITERATION CONVERGENCE DEMONSTRATION")
    print("="*70)
    print("This demo shows that policy iteration converges to the optimal policy")
    print("regardless of the initial policy choice.")
    print()

    env = create_classic_gridworld_env(formula_type=formula_type)

    # Test different initial policies
    test_policies = [
        ("Always Right", env.create_directional_policy('RIGHT')),
        ("Always Left", env.create_directional_policy('LEFT')),
        ("Always Up", env.create_directional_policy('UP')),
        ("Always Down", env.create_directional_policy('DOWN')),
        ("Clockwise", env.create_clockwise_policy()),
        ("Random (seed=42)", env.create_random_policy(seed=42)),
    ]

    results = []

    for policy_name, initial_policy in test_policies:
        print(f"\n{'-'*50}")
        print(f"Testing: {policy_name}")
        print(f"{'-'*50}")

        # Show the initial policy
        env.print_policy(initial_policy, f"Initial Policy: {policy_name}")

        # Set the initial policy and run policy iteration
        env.set_initial_policy(initial_policy)
        iterations = env.policy_iteration(max_iterations=10)

        # Get final policy and values
        final_policy = env.policy_history[-1]
        final_values = env.v_history[-1]

        print(f"Converged in {iterations} iterations")
        env.print_policy(final_policy, f"Final Optimal Policy")

        results.append({
            'name': policy_name,
            'iterations': iterations,
            'final_policy': final_policy.copy(),
            'final_values': final_values.copy()
        })

    # Compare final results
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Policy Name':<20} {'Iterations':<12} {'All Converged to Same?'}")
    print("-" * 70)

    # Check if all final policies are equivalent
    reference_policy = results[0]['final_policy']
    reference_values = results[0]['final_values']

    all_same_policy = True
    all_same_values = True

    for result in results:
        iterations = result['iterations']

        # Check policy equivalence
        same_policy = env._policies_equivalent(
            result['final_policy'], reference_policy)
        if not same_policy:
            all_same_policy = False

        # Check value similarity (within tolerance)
        same_values = np.allclose(
            result['final_values'], reference_values, atol=1e-3)
        if not same_values:
            all_same_values = False

        status = "✓" if same_policy and same_values else "✗"
        print(f"{result['name']:<20} {iterations:<12} {status}")

    print("-" * 70)
    print(
        f"All policies converged to optimal: {'✓ YES' if all_same_policy else '✗ NO'}")
    print(
        f"All values converged to optimal:  {'✓ YES' if all_same_values else '✗ NO'}")

    if all_same_policy and all_same_values:
        print("\n🎉 SUCCESS: All initial policies converged to the same optimal solution!")
        print("This demonstrates the fundamental property of policy iteration:")
        print("• Any policy will converge to the optimal policy")
        print("• The optimal policy is unique (up to ties)")
        print("• Policy iteration is guaranteed to find the global optimum")

    # Clear the custom policy for next use
    env.clear_initial_policy()

    return env, results


def demo_custom_policies():
    """
    Demonstrate how to create and use custom policies.
    """
    print("="*60)
    print("CUSTOM POLICY CREATION DEMO")
    print("="*60)

    env = create_classic_gridworld_env()

    print("Available policy creation methods:")
    print("1. create_directional_policy(direction)")
    print("2. create_random_policy(seed=None)")
    print("3. create_clockwise_policy()")
    print("4. Manual policy dictionary")
    print()

    # Demonstrate each type
    policies_to_show = [
        ("Always Right", env.create_directional_policy('RIGHT')),
        ("Always Left", env.create_directional_policy('LEFT')),
        ("Clockwise", env.create_clockwise_policy()),
        ("Random", env.create_random_policy(seed=123)),
    ]

    for name, policy in policies_to_show:
        env.print_policy(policy, f"{name} Policy")

    # Show how to create a manual policy
    print("Example: Creating a manual 'towards center' policy:")
    manual_policy = {}
    center_r, center_c = env.rows // 2, env.cols // 2

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if env.is_valid_state(r, c) and state not in env.terminals:
                # Move towards center
                if r < center_r:
                    action = 'D'
                elif r > center_r:
                    action = 'U'
                elif c < center_c:
                    action = 'R'
                else:
                    action = 'L'
                manual_policy[state] = action

    env.print_policy(manual_policy, "Manual 'Towards Center' Policy")

    print("\nTo use any of these policies:")
    print("1. policy = env.create_directional_policy('RIGHT')")
    print("2. env.set_initial_policy(policy)")
    print("3. env.policy_iteration()")
    print()

    return env


def demo_interactive_policy_testing():
    """
    Create an interactive demo for testing different initial policies.
    """
    print("="*60)
    print("INTERACTIVE POLICY TESTING")
    print("="*60)
    print("This demo allows you to interactively test different policies")
    print("and see how they converge using the visualization widgets.")
    print()

    env = create_classic_gridworld_env()

    print("Example usage:")
    print("1. Choose a starting policy:")
    print("   - right_policy = env.create_directional_policy('RIGHT')")
    print("   - env.set_initial_policy(right_policy)")
    print("2. Create visualization widgets:")
    print("   - widgets = env.create_visualization_widgets()")
    print("3. Select 'Policy Iteration' and click 'Run Algorithm'")
    print("4. Use the sliders to see convergence step by step")
    print()

    # Set up a default policy for immediate testing
    right_policy = env.create_directional_policy('RIGHT')
    env.set_initial_policy(right_policy)
    env.print_policy(right_policy, "Starting Policy: Always Right")

    print("A 'Always Right' policy has been set as default.")
    print("You can now create visualization widgets and test!")
    print()
    print("Try these other policies:")
    print("• env.set_initial_policy(env.create_directional_policy('LEFT'))")
    print("• env.set_initial_policy(env.create_clockwise_policy())")
    print("• env.set_initial_policy(env.create_random_policy(seed=42))")

    return env


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
    print(f"  Formula Type: {env.formula_type}")

    print("\n" + "="*60)
    print("CUSTOM INITIAL POLICY FEATURES (NEW!)")
    print("="*60)
    print("You can now test policy iteration with custom starting policies:")
    print()
    print("• Create directional policies:")
    print("  right_policy = env.create_directional_policy('RIGHT')")
    print("  env.set_initial_policy(right_policy)")
    print()
    print("• Create clockwise/pattern policies:")
    print("  clockwise = env.create_clockwise_policy()")
    print("  env.set_initial_policy(clockwise)")
    print()
    print("• Create random policies:")
    print("  random_policy = env.create_random_policy(seed=42)")
    print("  env.set_initial_policy(random_policy)")
    print()
    print("• View policies before running:")
    print("  env.print_policy(policy, 'My Policy')")
    print()
    print("• All policies converge to the same optimal solution!")
    print("  (Demonstrates the power of policy iteration)")
    print()

    print("="*60)
    print("INTERACTIVE VISUALIZATION GUIDE")
    print("="*60)
    print("VALUE ITERATION:")
    print("  - Single slider controls iteration")
    print("  - Shows complete convergence steps")
    print()
    print("POLICY ITERATION (ENHANCED!):")
    print("  - Two sliders for better understanding:")
    print("    1. 'Policy Iter': Which policy we're working with")
    print("    2. 'Eval Step': Policy evaluation convergence steps")
    print("  - Purple arrows = fixed policy during evaluation")
    print("  - See how values converge for each policy!")
    print("  - Watch policy evaluation → policy improvement cycle")
    print("  - Test different starting policies!")
    print()
    print("CONTROLS:")
    print("  - 'Run Algorithm': Execute selected algorithm")
    print("  - 'Next Step': Advance one step (smart stepping)")
    print("  - Checkboxes: Toggle display options")
    print("="*60)

    # Set up a demo policy for immediate testing
    print("\nSetting up demo: 'Always Right' initial policy...")
    right_policy = env.create_directional_policy('RIGHT')
    env.set_initial_policy(right_policy)
    env.print_policy(right_policy, "Demo Starting Policy")

    # Create and display widgets
    widgets = env.create_visualization_widgets()

    return env, widgets


if __name__ == "__main__":

    debug_value_iteration()

    debug_policy_iteration()

    # Test tie-breaking functionality (creates PNG files)
    print("\n" + "="*60)
    print("TESTING TIE-BREAKING FUNCTIONALITY")
    print("When Q-values are tied, multiple arrows will be shown overlapping")
    print("="*60)
    test_ties()
    test_extreme_ties()