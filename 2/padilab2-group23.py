import numpy as np

# Define state space dimensions
B_STATES = 5 # for B = 0, 1, 2, 3, 4
W_STATES = 5 # for W = 0, 1, 2
C_STATES = 3 # for C = 0, 1, 2, 3, 4

# Total number of regular states
NUM_REGULAR_STATES = B_STATES * W_STATES * C_STATES # 5 * 3 * 5 = 75

# Total number of states including the failure state (index 75)
NUM_TOTAL_STATES = NUM_REGULAR_STATES + 1 # 75 + 1 = 76

# Number of actions
NUM_ACTIONS = 3

# Define action indices
AC = 0 # Action Charge
AS = 1 # Action Solar
AE = 2 # Action Export


def factors_to_state(b, w, c):
    """
    Converts (b, w, c) factors to a unique state index.
    Raises ValueError if factors are outside defined ranges.
    """
    if not (0 <= b < B_STATES and 0 <= w < W_STATES and 0 <= c < C_STATES):
        raise ValueError(f"Invalid factors: b={b}, w={w}, c={c} must be within defined space dimensions.")
        
    return b * (W_STATES * C_STATES) + w * C_STATES + c

def state_to_factors(state_index):
    """
    Converts a state index to (b, w, c) factors.
    Handles the failure state (index NUM_REGULAR_STATES) by returning (-1, -1, -1).
    Raises ValueError for invalid regular state indices.
    """
    if state_index == NUM_REGULAR_STATES: # State 75 is the failure state
        return (-1, -1, -1) # Indicating a non-factorable failure state
    
    if not (0 <= state_index < NUM_REGULAR_STATES):
        raise ValueError(f"Invalid state_index: {state_index}. Must be between 0 and {NUM_TOTAL_STATES}")
        
    b = state_index // (W_STATES * C_STATES)
    remaining_index = state_index % (W_STATES * C_STATES)
    w = remaining_index // C_STATES
    c = remaining_index % C_STATES
    return (b, w, c)

def create_complex_transition_matrix():
    """
    Creates and populates the 3D transition tensor (T) with probabilistic transitions.
    T has shape (NUM_ACTIONS, NUM_TOTAL_STATES, NUM_TOTAL_STATES).
    """
    T = np.zeros((NUM_ACTIONS, NUM_TOTAL_STATES, NUM_TOTAL_STATES))
    failure_state_index = NUM_TOTAL_STATES - 1
    
    # Weather Transition Matrix P(W_next | W_current)
    # P_W[W_current, W_next]
    P_W = np.zeros((W_STATES, W_STATES))
    P_W[0, 1] = 0.3
    P_W[0, 0] = 0.7
    P_W[1, 2] = 0.4
    P_W[1, 0] = 0.3
    P_W[1, 1] = 0.3
    P_W[2, 3] = 0.4
    P_W[2, 2] = 0.3
    P_W[2, 1] = 0.3
    P_W[3, 4] = 0.4
    P_W[3, 3] = 0.3
    P_W[3, 2] = 0.3
    P_W[4, 4] = 0.7
    P_W[4, 3] = 0.3
    
    # Consumption Probability Function P(C_next | W_next)
    def get_prob_c_next(c_next_val, w_next_val):
        if w_next_val in [0, 1]: # If W_next is 1 or 2, C uniform over [1, 2, 3, 4]
            if c_next_val >= 1 and c_next_val <= 2:
                return 0.5
            else:
                return 0.0
        elif w_next_val == 2: # If W_next is 0, C uniform over [0, 1, 2]
            if c_next_val >= 0 and c_next_val <= 2:
                return 1 / 3
            else:
                return 0.0
        else:  # w_next_val in [3, 4]
            if c_next_val >= 0 and c_next_val <= 1:
                return 0.5
            else:
                return 0.0

    # Failure state transitions to itself for all actions with probability 1
    for action_idx in range(NUM_ACTIONS):
        T[action_idx, failure_state_index, failure_state_index] = 1.0

    # Populate transitions for regular states (0 to NUM_REGULAR_STATES - 1)
    for s_current in range(NUM_REGULAR_STATES):
        b_current, w_current, c_current = state_to_factors(s_current)

        # --- Handle Action 0 (Charge) ---
        action_idx = AC

        # Calculate deterministic B_next for normal charge outcome
        b_next_AC_normal = min(max(b_current - c_current, 0) + 1, B_STATES - 1)

        # Distribute the remaining probability (1 - prob_failure_AC) over possible W_next and C_next
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val)
                
                if prob_w_c_transition > 0:
                    s_next = factors_to_state(b_next_AC_normal, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next] += prob_w_c_transition


        # --- Handle Action 1 (Solar - AS) ---
        action_idx = AS
        failure_prob_AS = 0.0
        
        # Determine charging probability based on current weather
        charge_prob_AS = 0.0
        
        if w_current == 4:
            charge_prob_AS = 0.95
        elif w_current == 3:
            charge_prob_AS = 0.8
        elif w_current == 2:
            charge_prob_AS = 0.5
        elif w_current == 1:
            charge_prob_AS = 0.2
        elif w_current == 0:
            charge_prob_AS = 0.05
            failure_prob_AS = 0.02

        # Calculate deterministic B_next for charged and uncharged outcomes
        b_next_AS_uncharged = max(b_current - c_current, 0)
        b_next_AS_charged = min(b_next_AS_uncharged + 1, B_STATES - 1)

        # Distribute probabilities over possible W_next and C_next for both outcomes
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val)
                
                if prob_w_c_transition > 0:
                    # Add probability for the 'charged' outcome
                    s_next_charged = factors_to_state(b_next_AS_charged, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next_charged] += charge_prob_AS * prob_w_c_transition

                    # Add probability for the 'uncharged' outcome
                    s_next_uncharged = factors_to_state(b_next_AS_uncharged, w_next_val, c_next_val)
                    T[action_idx, s_current, s_next_uncharged] += (1 - charge_prob_AS - failure_prob_AS) * prob_w_c_transition

                    # Add probability for the 'failure' outcome
                    T[action_idx, s_current, failure_state_index] += failure_prob_AS * prob_w_c_transition


        # --- Handle Action 2 (Export - AE) ---
        action_idx = AE
        failure_prob_AE = 0.0
        
        # Calculate deterministic B_next based on current B and C
        b_next_AE = max(b_current - c_current - 1, 0)

        if b_next_AE == 0:
            failure_prob_AE = 0.05

        T[action_idx, s_current, failure_state_index] += failure_prob_AE

        # Distribute probabilities over possible W_next and C_next
        for w_next_val in range(W_STATES):
            for c_next_val in range(C_STATES):
                prob_w_c_transition = P_W[w_current, w_next_val] * get_prob_c_next(c_next_val, w_next_val) * (1 - failure_prob_AE)
                
                if prob_w_c_transition > 0:
                    s_next = factors_to_state(int(b_next_AE), w_next_val, c_next_val)
                    T[action_idx, s_current, s_next] += prob_w_c_transition
                    

    for a in range(T.shape[0]):
        for s in range(T.shape[1]):
            assert np.all(np.isclose(T[a, s].sum(), 1.0)), (s, a, T[a, s].sum())
            
    return T

# --- Main execution --- 
# Create the complex transition matrix
T_complex = create_complex_transition_matrix()

def print_transition_examples(action_idx, action_name, max_initial_states_to_show=5, max_transitions_per_initial_state=3):
    print(f"\n--- Examples for Action {action_idx} ({action_name}) ---")
    
    # Define some interesting example states (b, w, c) to showcase different scenarios
    example_states = [
        (0, 0, 0),   # Low battery, no sun, no consumption
        (4, 2, 0),   # Full battery, sunny, no consumption
        (1, 4, 0),   # Low battery, no sun, high consumption (potential for 0 battery)
        (3, 1, 2),   # Medium battery, cloudy, medium consumption
        (0, 4, 2)    # Empty battery, sunny, high consumption (challenging state)
    ]
    
    states_shown_count = 0
    
    for b_cur, w_cur, c_cur in example_states:
        if states_shown_count >= max_initial_states_to_show:
            break
            
        s_current = factors_to_state(b_cur, w_cur, c_cur)
        print(f"  From Initial State {s_current} ({b_cur},{w_cur},{c_cur}):")
        transitions_for_state = []
        
        for s_next in range(NUM_TOTAL_STATES):
            prob = T_complex[action_idx, s_current, s_next]
            
            if prob > 0:
                transitions_for_state.append((prob, s_next))

        # Sort by probability descending
        transitions_for_state.sort(key=lambda x: x[0], reverse=True)
        transitions_printed_for_this_state = 0
        
        for prob, s_next in transitions_for_state:
            if transitions_printed_for_this_state >= max_transitions_per_initial_state:
                break
            
            if s_next == NUM_TOTAL_STATES - 1: # Failure state
                print(f"    to Failure State {s_next} with P = {prob:.4f}")
            else:
                b_next, w_next, c_next = state_to_factors(s_next)
                print(f"    to State {s_next} ({b_next},{w_next},{c_next}) with P = {prob:.4f}")
                
            transitions_printed_for_this_state += 1
        
        if transitions_printed_for_this_state == 0:
            print("    No transitions with P > 0 found (this should not happen for valid MDPs unless states are terminal).")
            
        states_shown_count += 1

# Print examples for each action
print_transition_examples(AC, "Charge", max_initial_states_to_show=5, max_transitions_per_initial_state=3)
print_transition_examples(AS, "Solar", max_initial_states_to_show=5, max_transitions_per_initial_state=3)
print_transition_examples(AE, "Export", max_initial_states_to_show=5, max_transitions_per_initial_state=3)

# Demonstrate state conversion functions with examples (unchanged from before)
print("\n--- Demonstrating state conversion functions (unchanged) ---")
print(f"Factors (0, 0, 0) -> State: {factors_to_state(0, 0, 0)}")
print(f"Factors (4, 4, 2) -> State: {factors_to_state(4, 4, 2)}") # Max regular state
print(f"State 0 -> Factors: {state_to_factors(0)}")
print(f"State 74 -> Factors: {state_to_factors(74)}") # Max regular state
print(f"State 75 (Failure) -> Factors: {state_to_factors(75)}") # Failure state

import matplotlib.pyplot as plt

# Discount factor
gamma = 0.9

C = np.zeros((NUM_ACTIONS, NUM_TOTAL_STATES))

# Failure state inherits cost 500 regardless of action
C[:, NUM_REGULAR_STATES] = 500

for s in range(NUM_REGULAR_STATES):
    b, w, c_val = state_to_factors(s)

    # Action Charge (AC): always buys 1 unit from grid
    C[AC, s] = 10

    # Action Solar (AS)
    C[AS, s] = 0

    # Action Export (AE): sells 1 unit of stored energy to grid
    C[AE, s] = -5

# --- Value Iteration ---
def value_iteration(T, C, gamma, tol=1e-8, max_iter=5000):
    V = np.zeros(NUM_TOTAL_STATES)
    history = {"bellman_error": [], "policy_changes": []}
    prev_policy = np.zeros(NUM_TOTAL_STATES, dtype=int)

    for k in range(max_iter):
        Q = C + gamma * np.tensordot(T, V, axes=([2], [0]))  # shape (3, NUM_TOTAL_STATES)
        V_new = np.min(Q, axis=0)
        policy = np.argmin(Q, axis=0)

        bellman_error = np.max(np.abs(V_new - V))
        policy_changes = np.sum(policy != prev_policy)

        history["bellman_error"].append(bellman_error)
        history["policy_changes"].append(policy_changes)

        prev_policy = policy.copy()
        V = V_new

        if bellman_error < tol:
            print(f"Value Iteration converged in {k + 1} iterations (Bellman error < {tol})")
            break
    else:
        print(f"Value Iteration did not converge within {max_iter} iterations")

    return V, policy, history

# --- Policy Iteration ---
def policy_iteration(T, C, gamma, max_iter=500):
    policy = np.zeros(NUM_TOTAL_STATES, dtype=int)
    history = {"bellman_error": [], "policy_changes": []}

    for k in range(max_iter):
        # Policy evaluation
        T_pi = np.zeros((NUM_TOTAL_STATES, NUM_TOTAL_STATES))
        C_pi = np.zeros(NUM_TOTAL_STATES)
        for s in range(NUM_TOTAL_STATES):
            a = policy[s]
            T_pi[s, :] = T[a, s, :]
            C_pi[s] = C[a, s]

        V = np.linalg.solve(np.eye(NUM_TOTAL_STATES) - gamma * T_pi, C_pi)

        # Policy improvement
        Q = C + gamma * np.tensordot(T, V, axes=([2], [0]))
        new_policy = np.argmin(Q, axis=0)

        policy_changes = np.sum(new_policy != policy)

        if k == 0:
            V_old = np.zeros(NUM_TOTAL_STATES)
        bellman_error = np.max(np.abs(V - V_old))
        V_old = V.copy()

        history["bellman_error"].append(bellman_error)
        history["policy_changes"].append(policy_changes)

        if policy_changes == 0:
            print(f"Policy Iteration converged in {k + 1} iterations (policy stable)")
            break

        policy = new_policy
    else:
        print(f"Policy Iteration did not converge within {max_iter} iterations")

    return V, policy, history

# --- Run both algorithms and display clear outputs ---
print("=" * 60)
V_vi, pi_vi, hist_vi = value_iteration(T_complex, C, gamma)
print("=" * 60)
V_pi, pi_pi, hist_pi = policy_iteration(T_complex, C, gamma)
print("=" * 60)

# Comparison summary
print(f"\nMax |V_VI - V_PI| = {np.max(np.abs(V_vi - V_pi)):.2e}")
print(f"Policies agree on {np.sum(pi_vi == pi_pi)}/{NUM_TOTAL_STATES} states")
print(f"VI iterations: {len(hist_vi['bellman_error'])}")
print(f"PI iterations: {len(hist_pi['bellman_error'])}")

# Print converged value vectors (compact) and policies for readability
np.set_printoptions(precision=4, suppress=True)
print("\nValue Iteration converged V (first 20 states):")
print(V_vi[:20])
print("\nPolicy Iteration converged V (first 20 states):")
print(V_pi[:20])

action_names = {AC: 'Charge', AS: 'Solar', AE: 'Export'}
print("\nSample of converged policies (first 20 states):")
for s in range(min(20, NUM_TOTAL_STATES)):
    coord = 'Failure' if s == NUM_TOTAL_STATES - 1 else state_to_factors(s)
    print(f"s={s:2d} {str(coord):12} VI={action_names[int(pi_vi[s])]:6} PI={action_names[int(pi_pi[s])]:6} V_vi={V_vi[s]:7.2f} V_pi={V_pi[s]:7.2f}")

# Full vectors are also available as V_vi and V_pi
print('\n(Full converged vectors are stored in variables `V_vi` and `V_pi`.)' )


# ============================================================
# Cell 4: Impact of AC (Grid Charge) Cost on Value Iteration
# ============================================================
# AC costs to sweep; AS=0, AE=-5, Rfail=500 kept from cell 3
ac_costs = [0, 5, 32, 49, 75]

def compute_mttf(T, policy, cond_threshold=1e10):
    """
    Mean steps to failure via (I - T_pi_sub) h = ones.
    Returns np.inf when failure is unreachable under the policy
    (ill-conditioned or singular linear system).
    """
    n = NUM_REGULAR_STATES
    T_pi_sub = np.array([T[policy[s], s, :n] for s in range(n)])
    A = np.eye(n) - T_pi_sub
    if np.linalg.cond(A) > cond_threshold:
        return np.full(n, np.inf)
    try:
        h = np.linalg.solve(A, np.ones(n))
        return np.where(h > 0, h, np.inf)
    except np.linalg.LinAlgError:
        return np.full(n, np.inf)

action_names_local = {AC: 'Charge', AS: 'Solar', AE: 'Export'}
states_of_interest  = [0, 5, 32, 49, 75]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

sweep_results = {}

print("=" * 72)
print("  VALUE ITERATION  –  Sweeping AC cost  (AS=0, AE=-5, Rfail=500)")
print("=" * 72)

for ac_val in ac_costs:
    C_sweep = np.zeros((NUM_ACTIONS, NUM_TOTAL_STATES))
    C_sweep[:, NUM_REGULAR_STATES] = 500
    for s in range(NUM_REGULAR_STATES):
        C_sweep[AC, s] = ac_val
        C_sweep[AS, s] = 0
        C_sweep[AE, s] = -5

    V, policy, hist = value_iteration(T_complex, C_sweep, gamma)
    h         = compute_mttf(T_complex, policy)
    finite_h  = h[np.isfinite(h)]
    mean_mttf = float(np.mean(finite_h)) if finite_h.size > 0 else np.inf
    n_iter    = len(hist["bellman_error"])

    act_dist = {AC: 0, AS: 0, AE: 0}
    for a in policy[:NUM_REGULAR_STATES]:
        act_dist[int(a)] += 1

    sweep_results[ac_val] = {
        "C": C_sweep, "V": V, "policy": policy,
        "history": hist, "mttf": h, "mean_mttf": mean_mttf,
        "act_dist": act_dist,
    }

    mttf_str = "∞  (failure unreachable)" if np.isinf(mean_mttf) else f"{mean_mttf:.2f} steps"

    print(f"\n{'─'*72}")
    print(f"  AC cost = {ac_val}")
    print(f"{'─'*72}")
    print(f"  Convergence  : {n_iter} iterations")
    print(f"  Mean MTTF    : {mttf_str}")
    print(f"  Policy dist  : Charge={act_dist[AC]}, Solar={act_dist[AS]}, Export={act_dist[AE]}")

    print(f"\n  Policy & value at selected states:")
    print(f"  {'State':>6}  {'(b,w,c)':>12}  {'Action':>8}  {'V(s)':>10}  {'MTTF(s)':>12}")
    print(f"  {'':─>6}  {'':─>12}  {'':─>8}  {'':─>10}  {'':─>12}")
    for s in states_of_interest:
        fac   = state_to_factors(s)
        act   = action_names_local[int(policy[s])] if s < NUM_REGULAR_STATES else "—"
        v_val = f"{V[s]:.3f}"
        mttf_s = "—" if s >= NUM_REGULAR_STATES else ("∞" if np.isinf(h[s]) else f"{h[s]:.2f}")
        print(f"  {s:>6}  {str(fac):>12}  {act:>8}  {v_val:>10}  {mttf_s:>12}")

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print(f"\n{'='*72}")
print("  SUMMARY")
print(f"{'='*72}")
print(f"  {'AC cost':>8}  {'Iterations':>12}  {'Mean MTTF':>26}  "
      f"{'#Charge':>8}  {'#Solar':>7}  {'#Export':>8}")
print(f"  {'':─>8}  {'':─>12}  {'':─>26}  {'':─>8}  {'':─>7}  {'':─>8}")
for ac_val, r in sweep_results.items():
    mttf_str = "∞ (failure unreachable)" if np.isinf(r["mean_mttf"]) else f"{r['mean_mttf']:.2f}"
    ad = r["act_dist"]
    print(f"  {ac_val:>8}  {len(r['history']['bellman_error']):>12}  "
          f"{mttf_str:>26}  {ad[AC]:>8}  {ad[AS]:>7}  {ad[AE]:>8}")

# -------------------------------------------------------
# Plots: Convergence | Policy Changes | Policy Distribution
# -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Value Iteration – Impact of AC (Grid Charge) Cost", fontsize=13, fontweight="bold")

for (ac_val, r), color in zip(sweep_results.items(), colors):
    label = f"AC={ac_val}"
    axes[0].semilogy(r["history"]["bellman_error"], label=label, color=color)
    axes[1].plot(r["history"]["policy_changes"],    label=label, color=color)

axes[0].set_title("Bellman Error Convergence")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Max |ΔV|  (log scale)")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title("Policy Changes per Iteration")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("# State-action changes")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

# Stacked bar: policy distribution per AC cost
x        = np.arange(len(ac_costs))
n_charge = [sweep_results[k]["act_dist"][AC] for k in ac_costs]
n_solar  = [sweep_results[k]["act_dist"][AS] for k in ac_costs]
n_export = [sweep_results[k]["act_dist"][AE] for k in ac_costs]

axes[2].bar(x, n_charge, label="Charge (AC)", color="#1f77b4")
axes[2].bar(x, n_solar,  bottom=n_charge, label="Solar (AS)", color="#2ca02c")
axes[2].bar(x, n_export, bottom=[c+s for c,s in zip(n_charge, n_solar)],
            label="Export (AE)", color="#d62728")
axes[2].set_title("Policy Distribution across States\n(MTTF = ∞ for all AC costs)")
axes[2].set_xlabel("AC cost"); axes[2].set_ylabel("# States")
axes[2].set_xticks(x); axes[2].set_xticklabels([str(k) for k in ac_costs])
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()


# ── 1. Stationary distribution of the weather chain ──────────────────────────
P_W_mat = np.array([
    [0.7, 0.3, 0.0, 0.0, 0.0],
    [0.3, 0.3, 0.4, 0.0, 0.0],
    [0.0, 0.3, 0.3, 0.4, 0.0],
    [0.0, 0.0, 0.3, 0.3, 0.4],
    [0.0, 0.0, 0.0, 0.3, 0.7],
])
evals, evecs = np.linalg.eig(P_W_mat.T)
pi_w = np.abs(evecs[:, np.argmin(np.abs(evals - 1.0))])
pi_w /= pi_w.sum()
print(f"Weather stationary distribution π(w): {np.round(pi_w, 4)}")

# ── 2. Reduced (B × C) state space ───────────────────────────────────────────
NUM_REGULAR_MARG = B_STATES * C_STATES   # 15
NUM_TOTAL_MARG   = NUM_REGULAR_MARG + 1  # 16  (index 15 = failure)
FAILURE_MARG     = NUM_REGULAR_MARG

def bc_to_marg(b, c):
    return b * C_STATES + c

def marg_to_bc(s):
    if s == FAILURE_MARG:
        return (-1, -1)
    return (s // C_STATES, s % C_STATES)

# ── 3. Build marginalized transition tensor ───────────────────────────────────
T_marg = np.zeros((NUM_ACTIONS, NUM_TOTAL_MARG, NUM_TOTAL_MARG))
T_marg[:, FAILURE_MARG, FAILURE_MARG] = 1.0  # failure is absorbing

for s_bc in range(NUM_REGULAR_MARG):
    b, c_v = marg_to_bc(s_bc)
    for action in range(NUM_ACTIONS):
        for s_bc_next in range(NUM_TOTAL_MARG):
            prob = 0.0
            for w in range(W_STATES):
                s_full = factors_to_state(b, w, c_v)
                if s_bc_next == FAILURE_MARG:
                    prob += pi_w[w] * T_complex[action, s_full, NUM_REGULAR_STATES]
                else:
                    b_n, c_n = marg_to_bc(s_bc_next)
                    for w_n in range(W_STATES):
                        s_full_next = factors_to_state(b_n, w_n, c_n)
                        prob += pi_w[w] * T_complex[action, s_full, s_full_next]
            T_marg[action, s_bc, s_bc_next] = prob

for a in range(NUM_ACTIONS):
    for s in range(NUM_TOTAL_MARG):
        assert np.isclose(T_marg[a, s].sum(), 1.0), f"Row sum error a={a} s={s}"
print("T_marg row-sum check passed.")

# ── 4. Build marginalized cost matrix ────────────────────────────────────────
# C_full does not depend on w here (AC=10, AS=0, AE=-5) so marginalisation
# leaves the values unchanged.
C_marg = np.zeros((NUM_ACTIONS, NUM_TOTAL_MARG))
C_marg[:, FAILURE_MARG] = 500
for s_bc in range(NUM_REGULAR_MARG):
    C_marg[AC, s_bc] = 10
    C_marg[AS, s_bc] = 0
    C_marg[AE, s_bc] = -5

# ── 5. VI and PI helpers for arbitrary state-space size ──────────────────────
def value_iteration_gen(T, C, gamma, n_states, tol=1e-8, max_iter=5000):
    V = np.zeros(n_states)
    history = {"bellman_error": [], "policy_changes": []}
    prev_policy = np.zeros(n_states, dtype=int)
    for k in range(max_iter):
        Q = C + gamma * np.tensordot(T, V, axes=([2], [0]))
        V_new = np.min(Q, axis=0)
        policy = np.argmin(Q, axis=0)
        be = np.max(np.abs(V_new - V))
        pc = np.sum(policy != prev_policy)
        history["bellman_error"].append(be)
        history["policy_changes"].append(pc)
        prev_policy = policy.copy()
        V = V_new
        if be < tol:
            print(f"  VI converged in {k+1} iterations")
            break
    else:
        print("  VI did not converge")
    return V, policy, history

def policy_iteration_gen(T, C, gamma, n_states, max_iter=500):
    policy = np.zeros(n_states, dtype=int)
    history = {"bellman_error": [], "policy_changes": []}
    V_old = np.zeros(n_states)
    for k in range(max_iter):
        T_pi = np.array([T[policy[s], s, :] for s in range(n_states)])
        C_pi = np.array([C[policy[s], s]    for s in range(n_states)])
        V = np.linalg.solve(np.eye(n_states) - gamma * T_pi, C_pi)
        Q = C + gamma * np.tensordot(T, V, axes=([2], [0]))
        new_policy = np.argmin(Q, axis=0)
        pc = np.sum(new_policy != policy)
        be = np.max(np.abs(V - V_old))
        V_old = V.copy()
        history["bellman_error"].append(be)
        history["policy_changes"].append(pc)
        if pc == 0:
            print(f"  PI converged in {k+1} iterations")
            break
        policy = new_policy
    else:
        print("  PI did not converge")
    return V, policy, history

# ── 6. Run algorithms ─────────────────────────────────────────────────────────
print("\n--- Marginalized MDP (no weather, 16 states) ---")
print("=" * 60)
V_marg_vi, pi_marg_vi, hist_marg_vi = value_iteration_gen(
    T_marg, C_marg, gamma, NUM_TOTAL_MARG)
print("=" * 60)
V_marg_pi, pi_marg_pi, hist_marg_pi = policy_iteration_gen(
    T_marg, C_marg, gamma, NUM_TOTAL_MARG)
print("=" * 60)
print(f"VI/PI policies agree on {np.sum(pi_marg_vi == pi_marg_pi)}/{NUM_TOTAL_MARG} states")

# ── 7. Print marginalized policy ─────────────────────────────────────────────
anames = {AC: 'Charge', AS: 'Solar', AE: 'Export'}

print("\nMarginalized policy (B × C states, weather removed):")
print(f"  {'s':>4}  {'(b,c)':>7}  {'VI action':>10}  {'V_vi':>8}  {'PI action':>10}  {'V_pi':>8}")
print(f"  {'':─>4}  {'':─>7}  {'':─>10}  {'':─>8}  {'':─>10}  {'':─>8}")
for s in range(NUM_TOTAL_MARG):
    bc   = marg_to_bc(s)
    a_vi = anames[int(pi_marg_vi[s])] if s < NUM_REGULAR_MARG else "Failure"
    a_pi = anames[int(pi_marg_pi[s])] if s < NUM_REGULAR_MARG else "Failure"
    print(f"  {s:>4}  {str(bc):>7}  {a_vi:>10}  {V_marg_vi[s]:>8.3f}  {a_pi:>10}  {V_marg_pi[s]:>8.3f}")

# ── 8. Compare full vs marginalized per (b,c) pair ───────────────────────────
print("\nPolicy comparison: full model (by weather) vs marginalized model")
print(f"  {'(b,c)':>7}  {'w=0':>8} {'w=1':>8} {'w=2':>8} {'w=3':>8} {'w=4':>8}  {'Marg':>8}  note")
print(f"  {'':─>7}  {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8}  {'':─>8}")
for b in range(B_STATES):
    for c_v in range(C_STATES):
        s_marg = bc_to_marg(b, c_v)
        full_actions = [anames[int(pi_vi[factors_to_state(b, w, c_v)])][:3]
                        for w in range(W_STATES)]
        marg_act = anames[int(pi_marg_vi[s_marg])][:3]
        flag = "  ◄ weather matters" if len(set(full_actions)) > 1 else ""
        row = " ".join(f"{a:>8}" for a in full_actions)
        print(f"  {str((b,c_v)):>7}  {row}  {marg_act:>8}{flag}")

# ── 9. Convergence plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Full MDP (76 states, with weather)  vs  Marginalized MDP (16 states, no weather)",
             fontsize=12, fontweight="bold")

axes[0].semilogy(hist_vi["bellman_error"],      label="Full  – VI  (76 states)", color="#1f77b4")
axes[0].semilogy(hist_pi["bellman_error"],      label="Full  – PI  (76 states)", color="#1f77b4", linestyle="--")
axes[0].semilogy(hist_marg_vi["bellman_error"], label="Marg  – VI  (16 states)", color="#d62728")
axes[0].semilogy(hist_marg_pi["bellman_error"], label="Marg  – PI  (16 states)", color="#d62728", linestyle="--")
axes[0].set_title("Bellman Error Convergence")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Max |ΔV|  (log scale)")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_vi["policy_changes"],      label="Full – VI  (76 states)", color="#1f77b4")
axes[1].plot(hist_marg_vi["policy_changes"], label="Marg  – VI  (16 states)", color="#d62728")
axes[1].set_title("Policy Changes per VI Iteration")
axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("# State-action changes")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
