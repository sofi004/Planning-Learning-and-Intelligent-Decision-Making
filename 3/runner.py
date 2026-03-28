"""
Simplified Runner for Fishing Game AI
Trains 3 algorithms, produces a comparison, and saves the agents.
"""

from fishing_logic import FishingGameLogic, FISH_TYPES  # environment logic and fish presets
from agents import PredictiveAgent, QLearningAgent, SarsaLearningAgent  # agent implementations
# from lab3_agents_sol import PredictiveAgent, QLearningAgent, SarsaLearningAgent


def run_agent(
    agent,
    fish_types=None,
    num_episodes=2500,
    do_learning=True,
    verbose=True,
    visualize=False,
):
    """
    Run the agent in the environment for a set number of episodes.
    If do_learning=True, the agent will explore and update its policy.
    If do_learning=False, the agent will strictly exploit its current policy.
    """
    assert fish_types or num_episodes  # require a way to determine episode count

    if fish_types:
        num_episodes = len(fish_types)  # testing mode: number of episodes = len(fish_types)

    mode_str = "Training" if do_learning else "Testing"

    # Handle agent state for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)
    elif verbose:
        print(f"\n{'=' * 60}")
        print(f"{mode_str} {agent.__class__.__name__}")
        print(f"{'=' * 60}")

    visualizer = None

    if visualize:
        from visualize import GameVisualizer

        visualizer = GameVisualizer(
            f"{agent.__class__.__name__} ({mode_str})", "Random"
        )

    # bookkeeping
    wins = 0
    total_cost = 0
    total_steps = 0
    costs_history = []

    for episode in range(num_episodes):
        game = FishingGameLogic(fish_name=fish_types[episode] if fish_types else None)  # new episode env
        done = False
        episode_cost = 0
        steps = 0

        while not done and steps < 2500:
            state = game.get_state()  # current observation
            action = agent.get_action(state)  # agent decides

            next_state, cost, done = game.step_physics(action)  # env step
            next_action = agent.get_action(next_state)  # next action for on-policy learners

            if visualizer:
                visualizer.update(game.get_state(), done)  # optional GUI update

            if do_learning:
                agent.learn(state, action, cost, next_state, next_action, done)  # learning step

            episode_cost += cost
            steps += 1

        if do_learning:
            agent.end_episode()  # end-of-episode housekeeping

        costs_history.append(episode_cost)  # store episode cumulative cost
        total_cost += episode_cost
        total_steps += steps

        if game.catch_timer > 0:
            wins += 1  # count as a win if catch_timer positive at episode end

        # Print progress only when training and verbose
        if do_learning and verbose and (episode + 1) % 100 == 0:
            win_rate = wins / (episode + 1) * 100
            epsilon = getattr(agent, "epsilon", "N/A")
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Win Rate: {win_rate:5.1f}% | "
                f"ε: {epsilon if epsilon == 'N/A' else f'{epsilon:.3f}'}"
            )

    # Re-enable training if it was disabled for testing
    if not do_learning and hasattr(agent, "set_training_mode"):
        agent.set_training_mode(True)

    if visualizer:
        import time

        time.sleep(1)
        visualizer.close()

    win_rate = wins / num_episodes * 100
    avg_cost = total_cost / num_episodes
    avg_steps = total_steps / num_episodes

    return {
        "wins": wins,
        "win_rate": win_rate,
        "avg_cost": avg_cost,
        "avg_steps": avg_steps,
        "costs_history": costs_history,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Use classes instead of instances to re-initialize per run
    agent_classes = [
        #("PredictiveAgent", PredictiveAgent), # provides a baseline, it does not learn
        ("Q-Learning", QLearningAgent),
        # ("SARSA", SarsaLearningAgent),
    ]

    NUM_RUNS = 1  # number of repeated runs (averaging across repeats)
    NUM_TRAIN_EPISODES = 10000  # episodes used for training per run

    # Dictionaries to store cumulative summaries for plotting
    train_costs_all = {name: [] for name, _ in agent_classes}
    test_costs_all = {name: [] for name, _ in agent_classes}
    test_fish_types = [fish.name for fish in FISH_TYPES for _ in range(50)]  # long test set
    train_fish_types = ["Shark"] * NUM_TRAIN_EPISODES  # training curriculum (if used)
    results = {name: {"win_rate": [], "avg_cost": [], "avg_steps": []} for name, _ in agent_classes}

    for name, AgentClass in agent_classes:
        print(f"\n{'*' * 40}")
        print(f"Evaluating {name} over {NUM_RUNS} runs...")
        print(f"{'*' * 40}")

        for run in range(NUM_RUNS):
            print(f"--- {name} Run {run + 1}/{NUM_RUNS} ---")
            agent = AgentClass()

            # Train
            train_stats = run_agent(
                agent, 
                fish_types=None, 
                num_episodes=NUM_TRAIN_EPISODES, 
                do_learning=True, 
                verbose=True
            )

            if train_stats:
                train_costs_all[name].append(np.cumsum(train_stats["costs_history"]))

            # Test
            test_stats = run_agent(
                agent,
                fish_types=test_fish_types,
                num_episodes=None,
                do_learning=False,
                verbose=(run == 0),
                visualize=False, # disable per-episode GUI during automated runs
            )

            if test_stats:
                test_costs_all[name].append(np.cumsum(test_stats["costs_history"]))
                results[name]["win_rate"].append(test_stats["win_rate"])
                results[name]["avg_cost"].append(test_stats["avg_cost"])
                results[name]["avg_steps"].append(test_stats["avg_steps"])

    # Comparison
    print(f"\n{'=' * 60}")
    print("Comparison Summary (Average over multiple runs)")
    print(f"{'=' * 60}")
    print(f"{'Agent':<25} {'Win Rate':>10} {'Avg Cost':>12} {'Avg Steps':>11}")
    print("-" * 60)

    avg_results = []
    for name in results:
        if results[name]["win_rate"]:
            m_win = np.mean(results[name]["win_rate"])
            m_cost = np.mean(results[name]["avg_cost"])
            m_steps = np.mean(results[name]["avg_steps"])
            avg_results.append((name, m_win, m_cost, m_steps))
            print(f"{name:<25} {m_win:>9.1f}% {m_cost:>12.1f} {m_steps:>11.1f}")

    if avg_results:
        best_agent_info = max(avg_results, key=lambda x: x[1])
        print(f"\nBest Agent: {best_agent_info[0]} ({best_agent_info[1]:.1f}% mean win rate)")

    print("\nTraining and comparison completed.")

    # Plot cumulative cost summaries (mean ± std) for training and testing
    print("\nGenerating Cumulative Cost Plots (Mean ± Std Dev)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training plot
    for name, runs_data in train_costs_all.items():
        if not runs_data:
            continue
        runs_matrix = np.array(runs_data)
        mean_costs = np.mean(runs_matrix, axis=0)
        std_costs = np.std(runs_matrix, axis=0)
        p = ax1.plot(mean_costs, label=name)
        color = p[0].get_color()
        ax1.fill_between(range(len(mean_costs)), mean_costs - std_costs, mean_costs + std_costs, color=color, alpha=0.3)

    ax1.set_title("Training: Cumulative Cost")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Cost")
    ax1.legend()
    ax1.grid(True)

    # Testing plot
    for name, runs_data in test_costs_all.items():
        if not runs_data:
            continue
        runs_matrix = np.array(runs_data)
        mean_costs = np.mean(runs_matrix, axis=0)
        std_costs = np.std(runs_matrix, axis=0)
        p = ax2.plot(mean_costs, label=name)
        color = p[0].get_color()
        ax2.fill_between(range(len(mean_costs)), mean_costs - std_costs, mean_costs + std_costs, color=color, alpha=0.3)

    ax2.set_title("Testing: Cumulative Cost")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Cost")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Ensure output directory exists and save the combined figure
    os.makedirs("plots", exist_ok=True)
    fig_path = os.path.join("plots", "cumulative_costs.png")
    try:
        fig.savefig(fig_path)
        print(f"Saved plot to {os.path.abspath(fig_path)}")
    except Exception as e:
        print(f"Failed to save plot to {fig_path}: {e}")

    plt.show()
