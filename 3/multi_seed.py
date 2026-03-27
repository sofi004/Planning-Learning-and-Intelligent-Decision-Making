# multi_seed.py
import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from runner import run_agent
from agents import PredictiveAgent, QLearningAgent, SarsaLearningAgent
from fishing_logic import FISH_TYPES

# Seeds 1..5
SEEDS = [1, 2, 3, 4, 5]
NUM_TRAIN = 2500
TEST_FISH_TYPES = [f.name for f in FISH_TYPES for _ in range(50)]
AGENTS = [
          ("Q-Learning", QLearningAgent)]

os.makedirs("multi_results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Run seeds and save per-seed cumulative-cost CSVs
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for name, AgentClass in AGENTS:
        print(f"Seed {seed} — {name}: training")
        agent = AgentClass()
        train_stats = run_agent(agent, fish_types=None, num_episodes=NUM_TRAIN, do_learning=True, verbose=False, visualize=False)
        train_path = os.path.join("multi_results", f"{name}_seed{seed}_train_costs.csv")
        with open(train_path, "w", newline='') as f:
            f.write("episode,cum_cost\n")
            cum = 0.0
            for i, v in enumerate(train_stats["costs_history"], 1):
                cum += v
                f.write(f"{i},{cum}\n")
        print(f"Saved {train_path}")

        print(f"Seed {seed} — {name}: testing")
        test_stats = run_agent(agent, fish_types=TEST_FISH_TYPES, num_episodes=None, do_learning=False, verbose=False, visualize=False)
        test_path = os.path.join("multi_results", f"{name}_seed{seed}_test_costs.csv")
        with open(test_path, "w", newline='') as f:
            f.write("episode,cum_cost\n")
            cum = 0.0
            for i, v in enumerate(test_stats["costs_history"], 1):
                cum += v
                f.write(f"{i},{cum}\n")
        print(f"Saved {test_path}")


# Helper to read cumulative CSV into numpy array
def read_csv_cumulative(path):
    costs = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                costs.append(float(row[1]))
            except Exception:
                costs.append(0.0)
    return np.array(costs)


# Aggregate and make combined plots: one for training, one for testing
plt.style.use('seaborn-v0_8')
fig_t, ax_t = plt.subplots(1, 1, figsize=(10, 6))
fig_s, ax_s = plt.subplots(1, 1, figsize=(10, 6))

for name, _ in AGENTS:
    # collect train mats for seeds
    train_files = [os.path.join('multi_results', f"{name}_seed{seed}_train_costs.csv") for seed in SEEDS]
    train_mats = []
    for p in train_files:
        if os.path.exists(p):
            train_mats.append(read_csv_cumulative(p))
    if train_mats:
        maxlen = max(len(a) for a in train_mats)
        mat = np.zeros((len(train_mats), maxlen))
        for i, a in enumerate(train_mats):
            mat[i, : len(a)] = a
            if len(a) < maxlen:
                mat[i, len(a):] = a[-1] if len(a) > 0 else 0.0
        mean_t = np.mean(mat, axis=0)
        std_t = np.std(mat, axis=0)
        # plot individual runs
        for i in range(mat.shape[0]):
            ax_t.plot(mat[i], color='gray', alpha=0.3, linewidth=0.8)
        # plot mean and std
        ax_t.plot(mean_t, label=name, linewidth=2)
        ax_t.fill_between(range(len(mean_t)), mean_t - std_t, mean_t + std_t, alpha=0.2)
        # save aggregated CSV
        out_csv = os.path.join('results', f'{name}_train_mean_std.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'mean_cum', 'std_cum'])
            for i, (m, s) in enumerate(zip(mean_t, std_t), 1):
                w.writerow([i, m, s])

    # collect test mats for seeds
    test_files = [os.path.join('multi_results', f"{name}_seed{seed}_test_costs.csv") for seed in SEEDS]
    test_mats = []
    for p in test_files:
        if os.path.exists(p):
            test_mats.append(read_csv_cumulative(p))
    if test_mats:
        maxlen = max(len(a) for a in test_mats)
        mat = np.zeros((len(test_mats), maxlen))
        for i, a in enumerate(test_mats):
            mat[i, : len(a)] = a
            if len(a) < maxlen:
                mat[i, len(a):] = a[-1] if len(a) > 0 else 0.0
        mean_s = np.mean(mat, axis=0)
        std_s = np.std(mat, axis=0)
        # plot individual runs
        for i in range(mat.shape[0]):
            ax_s.plot(mat[i], color='gray', alpha=0.3, linewidth=0.8)
        # plot mean and std
        ax_s.plot(mean_s, label=name, linewidth=2)
        ax_s.fill_between(range(len(mean_s)), mean_s - std_s, mean_s + std_s, alpha=0.2)
        # save aggregated CSV
        out_csv = os.path.join('results', f'{name}_test_mean_std.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'mean_cum', 'std_cum'])
            for i, (m, s) in enumerate(zip(mean_s, std_s), 1):
                w.writerow([i, m, s])

# finalize training figure
ax_t.set_title('Training: Runs + Mean ± Std (Cumulative Cost)')
ax_t.set_xlabel('Episode')
ax_t.set_ylabel('Cumulative Cost')
ax_t.legend()
ax_t.grid(True)
p_train = os.path.abspath(os.path.join('plots', 'combined_training_mean_std.png'))
fig_t.tight_layout()
fig_t.savefig(p_train)
plt.close(fig_t)
print(f'Saved {p_train}')

# finalize testing figure
ax_s.set_title('Testing: Runs + Mean ± Std (Cumulative Cost)')
ax_s.set_xlabel('Episode')
ax_s.set_ylabel('Cumulative Cost')
ax_s.legend()
ax_s.grid(True)
p_test = os.path.abspath(os.path.join('plots', 'combined_testing_mean_std.png'))
fig_s.tight_layout()
fig_s.savefig(p_test)
plt.close(fig_s)
print(f'Saved {p_test}')
