import numpy as np
import matplotlib.pyplot as plt

####### EX1

NB = 5 
NW = 5
NC = 3 
N_OP = NB * NW * NC
N_TOTAL = N_OP + 1
FAILURE_IDX = N_OP


def idx(b, w, c):
    return (b * (NW * NC)) + (w * NC) + c


def inv_idx(index):
    if index == FAILURE_IDX:
        return (None, None, None)
    b = index // (NW * NC)
    rem = index % (NW * NC)
    w = rem // NC
    c = rem % NC
    return (b, w, c)


def clip(x, a, b):
    return max(a, min(b, x))


def _one_dim_transitions(value, max_value, stay_prob, move_prob):
    # Base assignments for neighbors and stay; then normalize across valid neighbors
    probs = {}
    if value > 0 and value != max_value:
        probs[value - 1] = move_prob
    elif value == 0:
        probs[value + 1] = move_prob * 2
    if value < max_value and value != 0:
        probs[value + 1] = move_prob
    elif value == max_value:
        probs[value - 1] = move_prob * 2
    
    probs[value] = stay_prob
    return probs


def build_matrix(scenario="A"):
    P = np.zeros((N_TOTAL, N_TOTAL), dtype=float)

    # For each operational state
    for b in range(NB):
        for w in range(NW):
            for c in range(NC):
                i = idx(b, w, c)

                # Physical battery update (deterministic)
                b_next = clip(b + w - c, 0, NB - 1)

                # Environmental transitions
                w_probs = _one_dim_transitions(w, NW - 1, stay_prob=0.6, move_prob=0.2)
                c_probs = _one_dim_transitions(c, NC - 1, stay_prob=0.7, move_prob=0.15)

                for w2, pw in w_probs.items():
                    for c2, pc in c_probs.items():
                        j = idx(b_next, w2, c2)
                        P[i, j] += pw * pc
    for i in range(N_OP):
        P[FAILURE_IDX, i] = 0
    P[FAILURE_IDX, FAILURE_IDX] = 1.0

    # Scenario B: brittle-bottom adjustment
    if scenario.upper() == "B":
        # total probability mass directed to next states with battery level 0
        for j in range(N_OP):
            if P[j,0] > 0.1:
                P[j,0] -= 0.1
                P[j,FAILURE_IDX] += 0.1

    if scenario.upper() == "C":
        for j in range(N_OP):
            b_next, _, _ = inv_idx(j)
            if b_next == 0 or b_next == NB - 1:
                for k in range(N_OP):
                    if P[j, k] > 0.05:
                        P[j, k] -= 0.05
                        P[j, FAILURE_IDX] += 0.05

    return P

####### EX2

def theoretical_failure_probs(P, s0, Ns):
    # initial distribution
    pi = np.zeros(P.shape[0])
    pi[s0] = 1.0
    mu = []
    for t in range(Ns + 1):
        mu.append(pi[FAILURE_IDX])
        pi = pi @ P
    return np.array(mu)


def simulate_trajectories(P, s0, Nr=5000, Ns=40, seed=0):
    rng = np.random.default_rng(seed)
    # precompute cumulative probabilities for faster sampling
    C = P.cumsum(axis=1)

    # current states for each trajectory
    curr = np.full(Nr, s0, dtype=int)
    frac_fail = [np.mean(curr == FAILURE_IDX)]

    for t in range(1, Ns + 1):
        r = rng.random(Nr)
        # sample next state for each trajectory
        next_states = np.empty_like(curr)
        for i in range(Nr):
            row = C[curr[i]]
            next_states[i] = np.searchsorted(row, r[i], side='right')
        curr = next_states
        frac_fail.append(np.mean(curr == FAILURE_IDX))

    return np.array(frac_fail)


def plot_compare(theory, empirical, out='failure_comparison.png'):
    Ns = len(theory) - 1
    t = np.arange(Ns + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, theory, label='Theoretical µ_t[75]', lw=2)
    plt.plot(t, empirical, label='Empirical fraction failed', lw=2, linestyle='--')
    plt.xlabel('Time step t')
    plt.ylabel('Failure probability / fraction')
    plt.title('Failure: theoretical vs empirical')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    print(f'Plot saved to {out}')

####### EX3
def power_iter(P, s0, N=1000):
    n = P.shape[0]
    if isinstance(s0, (list, tuple, np.ndarray)):
        mu = np.array(s0, dtype=float)
    else:
        mu = np.zeros(n, dtype=float)
        mu[s0] = 1.0

    for _ in range(N):
        mu = mu @ P
    return mu


def stationary_eig(P, tol=1e-12):
    vals, vecs = np.linalg.eig(P.T)
    # find eigenvalues close to 1
    close = np.isclose(vals, 1.0, atol=tol)
    if not np.any(close):
        raise RuntimeError("No eigenvalue 1 found")
    # pick the eigenvector corresponding to the eigenvalue 1 with largest real part
    idxs = np.where(close)[0]
    v = vecs[:, idxs[0]]
    v = np.real_if_close(v)
    v = v.astype(float)
    # normalize
    if v.sum() == 0:
        raise RuntimeError("Eigenvector sum is zero; cannot normalize")
    v = v / v.sum()
    # enforce non-negativity (numerical)
    v[v < 0] = 0.0
    v = v / v.sum()
    return v

######## EX4
def mean_time_to_absorption(P, absorbing_indices):
    n = P.shape[0]
    absorbing = set(absorbing_indices)
    transient = [i for i in range(n) if i not in absorbing]
    if len(transient) == 0:
        return np.zeros(n)
    Q = P[np.ix_(transient, transient)]
    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # singular: return inf for transient states
        t = np.full(n, np.inf)
        for a in absorbing:
            t[a] = 0.0
        return t
    ones = np.ones((Q.shape[0],))
    t_transient = N.dot(ones)
    t = np.full(n, 0.0)
    for idx, val in zip(transient, t_transient):
        t[idx] = val
    for a in absorbing:
        t[a] = 0.0
    return t


def mean_first_passage(P, target):
    n = P.shape[0]
    S = [i for i in range(n) if i != target]
    P_SS = P[np.ix_(S, S)]
    I = np.eye(P_SS.shape[0])
    ones = np.ones((P_SS.shape[0],))
    h = np.full(n, np.inf)
    try:
        x = np.linalg.solve(I - P_SS, ones)
    except np.linalg.LinAlgError:
        return h
    for idx, val in zip(S, x):
        h[idx] = val
    h[target] = 0.0
    return h
####### EX5

def build_matrix_both_stay(weather_stay, consume_stay, scenario='B'):
    """Build transition matrix varying both weather and consumption stay probabilities."""
    move_w = (1.0 - weather_stay) / 2.0
    move_c = (1.0 - consume_stay) / 2.0
    P = np.zeros((N_TOTAL, N_TOTAL), dtype=float)
    for b in range(NB):
        for w in range(NW):
            for c in range(NC):
                i = idx(b, w, c)
                b_next = clip(b + w - c, 0, NB - 1)
                # weather transitions
                w_probs = {}
                for dv, p in ((0, weather_stay), (-1, move_w), (1, move_w)):
                    w2 = w + dv
                    if 0 <= w2 <= NW - 1:
                        w_probs[w2] = w_probs.get(w2, 0.0) + p
                # consumption transitions with custom stay
                c_probs = {}
                for dv, p in ((0, consume_stay), (-1, move_c), (1, move_c)):
                    c2 = c + dv
                    if 0 <= c2 <= NC - 1:
                        c_probs[c2] = c_probs.get(c2, 0.0) + p
                for w2, pw in w_probs.items():
                    for c2, pc in c_probs.items():
                        j = idx(b_next, w2, c2)
                        P[i, j] += pw * pc
    P[FAILURE_IDX, :] = 0.0
    P[FAILURE_IDX, FAILURE_IDX] = 1.0
    if scenario.upper() == 'B':
        for i in range(N_OP):
            prob_to_b0 = 0.0
            b0_inds = []
            for j in range(N_OP):
                b_j, _, _ = inv_idx(j)
                if b_j == 0:
                    prob_to_b0 += P[i, j]
                    b0_inds.append(j)
            if prob_to_b0 > 0.1 and b0_inds:
                factor = (prob_to_b0 - 0.1) / prob_to_b0
                for j in b0_inds:
                    P[i, j] *= factor
                P[i, FAILURE_IDX] += 0.1
    row_sums = P.sum(axis=1)
    for i in range(N_TOTAL):
        if row_sums[i] > 0:
            P[i, :] /= row_sums[i]
    return P


def plot_failure_t100_vs_stays(ws_values=None, cs_values=None, s0=None, Ns=100,
                               out_png='failure_t100_vs_stays.png'):
    """Like the first panel of weather_sweep.png but at t=Ns, with one line per
    consumption stay probability. x-axis = weather stay prob."""
    if ws_values is None:
        ws_values = np.linspace(0.05, 0.95, 19)
    if cs_values is None:
        cs_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    if s0 is None:
        s0 = idx(2, 2, 1)

    plt.figure(figsize=(8, 5))
    for cs in cs_values:
        probs = []
        for ws in ws_values:
            P = build_matrix_both_stay(ws, cs, scenario='B')
            pi = np.zeros(P.shape[0]); pi[s0] = 1.0
            for _ in range(Ns):
                pi = pi @ P
            probs.append(pi[FAILURE_IDX])
        plt.plot(ws_values, probs, '-o', label=f'cons stay={cs:.1f}')

    plt.xlabel('Weather stay probability')
    plt.ylabel(f'Pr(failure at t={Ns})')
    plt.title(f'Pr(failure) at t={Ns}: weather vs consumption volatility (Scenario B)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f'Plot saved to {out_png}')


###### EX6:

def battery_level_comparison(b_values=None, scenario='C'):
    if b_values is None:
        b_values = [2, 4]
    P = build_matrix(scenario)
    t_abs = mean_time_to_absorption(P, [FAILURE_IDX])

    print(f"{'State':<16} {'P(fail,t=1)':>12} {'ETTF':>8} {'mu_10':>8} {'mu_40':>8} {'mu_100':>8}  B_next(W=2,C=1)")
    print('-' * 90)
    summary = {b: [] for b in b_values}
    for b in b_values:
        for w in range(NW):
            for c in range(NC):
                s = idx(b, w, c)
                # direct one-step failure probability from the matrix row
                p_direct = P[s, FAILURE_IDX]
                ettf = t_abs[s]
                # transient failure probs
                pi = np.zeros(P.shape[0]); pi[s] = 1.0
                vals = {}
                for t in range(1, 101):
                    pi = pi @ P
                    if t in (10, 40, 100):
                        vals[t] = pi[FAILURE_IDX]
                # b_next when W=2, C=1 as a representative case
                b_next_ex = clip(b + w - c, 0, NB - 1)
                boundary = b_next_ex in (0, NB - 1)
                print(f"(B={b},W={w},C={c})         {p_direct:>12.4f} {ettf:>8.2f} {vals[10]:>8.4f} {vals[40]:>8.4f} {vals[100]:>8.4f}  B'={b_next_ex} {'<BOUNDARY>' if boundary else ''}")
                summary[b].append(ettf)
        print()

    # aggregated comparison
    print('=== Summary ===')
    for b in b_values:
        ettfs = summary[b]
        print(f'B={b}  ETTF: min={min(ettfs):.2f}  max={max(ettfs):.2f}  mean={np.mean(ettfs):.2f}')


if __name__ == "__main__":
    print("Exercise 1")
    # quick smoke test and save
    P_A = build_matrix('A')
    P_B = build_matrix('B')
    P_C = build_matrix('C')
    print("P_A shape:", P_A.shape)
    print("P_B shape:", P_B.shape)
    print("P_C shape:", P_C.shape)
    print("P_A row-sum min/max:", P_A.sum(axis=1).min(), P_A.sum(axis=1).max())
    print("P_B row-sum min/max:", P_B.sum(axis=1).min(), P_B.sum(axis=1).max())
    print("P_C row-sum min/max:", P_C.sum(axis=1).min(), P_C.sum(axis=1).max())
    print("P_A :", P_A)
    print("P_B :", P_B)
    print("P_C :", P_C)

    print("Exercise 2")
    scenario = 'B'
    Ns = 40
    Nr = 5000
    # choose initial state s0: e.g., B=2, W=2, C=1
    s0 = idx(2, 2, 1)

    P = build_matrix(scenario)

    theory = theoretical_failure_probs(P, s0, Ns)
    empirical = simulate_trajectories(P, s0, Nr=Nr, Ns=Ns, seed=42)

    # print a few values
    for t in [0, 1, 5, 10, 20, Ns]:
        print(f't={t}: theory={theory[t]:.6f}, empirical={empirical[t]:.6f}')

    plot_compare(theory, empirical)
        
    print("Exercise 3")
    for N in [100, 1000, 5000, 20000]:
        muN = power_iter(P, s0, N=N)
        print(f'power_iter N={N}: mu_N[75] = {muN[75]:.6f}')

    # eigen-based stationary distribution
    pi = stationary_eig(P)
    print(f'stationary_eig[75] = {pi[75]:.6f}')

    # L1 distance between large-N power-iter result and eigen stationary
    mu_large = power_iter(P, s0, N=20000)
    l1 = np.linalg.norm(mu_large - pi, ord=1)
    print(f'L1 distance (mu_20000, pi) = {l1:.6e}')

    # multiplicity of eigenvalue 1 (diagnostic for uniqueness)
    mult = np.sum(np.isclose(np.linalg.eigvals(P.T), 1.0))
    print(f'eigenvalue-1 multiplicity: {int(mult)}')
    
    print("Exercise 4")
    # Scenario C: compute expected time to failure (absorption)
    P_C = build_matrix('C')

    for N in [100, 1000, 5000, 20000]:
        muN = power_iter(P_C, s0, N=N)
        print(f'power_iter N={N}: mu_N[75] = {muN[75]:.6f}')

    # eigen-based stationary distribution
    pi = stationary_eig(P_C)
    print(f'stationary_eig[75] = {pi[75]:.6f}')

    # L1 distance between large-N power-iter result and eigen stationary
    mu_large = power_iter(P_C, s0, N=20000)
    l1 = np.linalg.norm(mu_large - pi, ord=1)
    print(f'L1 distance (mu_20000, pi) = {l1:.6e}')

    # multiplicity of eigenvalue 1 (diagnostic for uniqueness)
    mult = np.sum(np.isclose(np.linalg.eigvals(P_C.T), 1.0))
    print(f'eigenvalue-1 multiplicity: {int(mult)}')

    t_abs = mean_time_to_absorption(P_C, [FAILURE_IDX])
    print(f'Expected time to failure from s0 (Scenario C): {t_abs[s0]:.4f} steps')

    # Mean first-passage time (s0 -> failure)
    h = mean_first_passage(P_C, FAILURE_IDX)
    print(f'MFPT (s0 -> failure) = {h[s0]:.4f} steps')

    # Example MFPT between two sample operational states
    s1 = idx(1, 1, 1)
    s2 = idx(3, 3, 1)
    h_s1_s2 = mean_first_passage(P_C, s2)
    print(f'MFPT (s1 -> s2) = {h_s1_s2[s1]:.4f} steps')

    print('\nExercise 5')
    plot_failure_t100_vs_stays()

    print('\nExercise 6')
    battery_level_comparison()


