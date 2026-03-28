import numpy as np  # numpy for array operations and linear algebra
import matplotlib.pyplot as plt  # plotting utilities from matplotlib

####### EX1  # marker for exercise 1 section

NB = 5  # number of battery levels
NW = 5  # number of weather states
NC = 3  # number of consumption states
N_OP = NB * NW * NC  # total number of operational states
N_TOTAL = N_OP + 1  # total states including the failure absorbing state
FAILURE_IDX = N_OP  # index used for the failure/absorbing state


def idx(b, w, c):  # map 3D state (b,w,c) to linear index
    return (b * (NW * NC)) + (w * NC) + c  # compute linearized index


def inv_idx(index):  # inverse of idx: linear index -> (b,w,c)
    if index == FAILURE_IDX:  # if index is failure
        return (None, None, None)  # failure has no (b,w,c)
    b = index // (NW * NC)  # battery component via integer division
    rem = index % (NW * NC)  # remainder after removing battery part
    w = rem // NC  # weather component
    c = rem % NC  # consumption component
    return (b, w, c)  # return tuple


def clip(x, a, b):  # clamp x into inclusive range [a,b]
    return max(a, min(b, x))  # return clamped value


def _one_dim_transitions(value, max_value, stay_prob, move_prob):  # 1D neighbour transition probs
    # Base assignments for neighbors and stay; then normalize across valid neighbors
    probs = {}  # dict mapping state -> probability mass
    if value > 0 and value != max_value:
        probs[value - 1] = move_prob  # probability to move down one
    elif value == 0:
        probs[value + 1] = move_prob * 2  # at lower bound shift mass up
    if value < max_value and value != 0:
        probs[value + 1] = move_prob  # probability to move up one
    elif value == max_value:
        probs[value - 1] = move_prob * 2  # at upper bound shift mass down
    
    probs[value] = stay_prob  # probability to stay in same state
    return probs  # return mapping


def build_matrix(scenario="A"):  # build full transition matrix for scenarios A/B/C
    P = np.zeros((N_TOTAL, N_TOTAL), dtype=float)  # initialize matrix with zeros

    # For each operational state
    for b in range(NB):  # iterate battery levels
        for w in range(NW):  # iterate weather states
            for c in range(NC):  # iterate consumption levels
                i = idx(b, w, c)  # linear index for current state

                # Physical battery update (deterministic)
                b_next = clip(b + w - c, 0, NB - 1)  # next battery after consumption/gain

                # Environmental transitions
                w_probs = _one_dim_transitions(w, NW - 1, stay_prob=0.6, move_prob=0.2)  # weather probs
                c_probs = _one_dim_transitions(c, NC - 1, stay_prob=0.7, move_prob=0.15)  # consumption probs

                for w2, pw in w_probs.items():  # combine weather outcomes
                    for c2, pc in c_probs.items():  # combine consumption outcomes
                        j = idx(b_next, w2, c2)  # resulting state index
                        P[i, j] += pw * pc  # accumulate joint probability
    for i in range(N_OP):  # for operational-state rows
        P[FAILURE_IDX, i] = 0  # failure row has zero to operational states
    P[FAILURE_IDX, FAILURE_IDX] = 1.0  # failure is absorbing

    # Scenario B: brittle-bottom adjustment
    if scenario.upper() == "B":  # if scenario B, reduce certain transitions to b=0
        # total probability mass directed to next states with battery level 0
        for j in range(N_OP):
            if P[j,0] > 0.1:
                P[j,0] -= 0.1  # remove 0.1 probability mass to b=0
                P[j,FAILURE_IDX] += 0.1  # send that mass to failure

    if scenario.upper() == "C":  # scenario C: penalize transitions from boundary battery states
        for j in range(N_OP):
            b_next, _, _ = inv_idx(j)  # get battery component of index j
            if b_next == 0 or b_next == NB - 1:  # if boundary battery
                for k in range(N_OP):
                    if P[j, k] > 0.05:
                        P[j, k] -= 0.05  # reduce some outgoing mass
                        P[j, FAILURE_IDX] += 0.05  # add it to failure

    return P  # return the constructed matrix

####### EX2  # marker for exercise 2

def theoretical_failure_probs(P, s0, Ns):  # compute exact failure probabilities µ_t up to Ns
    # initial distribution
    pi = np.zeros(P.shape[0])  # start with all-zero distribution
    pi[s0] = 1.0  # place mass at initial state s0
    mu = []  # list to collect failure probabilities over time
    for t in range(Ns + 1):  # include time 0
        mu.append(pi[FAILURE_IDX])  # record prob of failure at current time
        pi = pi @ P  # advance distribution by one timestep
    return np.array(mu)  # return as numpy array


def simulate_trajectories(P, s0, Nr=5000, Ns=40, seed=0):  # Monte-Carlo simulate Nr trajectories
    rng = np.random.default_rng(seed)  # RNG with seed for reproducibility
    # precompute cumulative probabilities for faster sampling
    C = P.cumsum(axis=1)  # cumulative sums per row for inverse transform sampling

    # current states for each trajectory
    curr = np.full(Nr, s0, dtype=int)  # initialize all trajectories at s0
    frac_fail = [np.mean(curr == FAILURE_IDX)]  # initial fraction failed (usually 0)

    for t in range(1, Ns + 1):  # simulate Ns steps
        r = rng.random(Nr)  # draw uniform randoms for each trajectory
        # sample next state for each trajectory
        next_states = np.empty_like(curr)  # array to store sampled next states
        for i in range(Nr):  # loop over trajectories (explicit for clarity)
            row = C[curr[i]]  # cumulative distribution for the current state's row
            next_states[i] = np.searchsorted(row, r[i], side='right')  # find next state
        curr = next_states  # update current states
        frac_fail.append(np.mean(curr == FAILURE_IDX))  # record fraction failed

    return np.array(frac_fail)  # return time series of empirical failure fractions


def plot_compare(theory, empirical, out='failure_comparison.png'):  # plot theoretical vs empirical
    Ns = len(theory) - 1  # number of steps
    t = np.arange(Ns + 1)  # time axis
    plt.figure(figsize=(8, 4.5))  # create figure
    plt.plot(t, theory, label='Theoretical µ_t[75]', lw=2)  # plot theory
    plt.plot(t, empirical, label='Empirical fraction failed', lw=2, linestyle='--')  # plot empirical
    plt.xlabel('Time step t')  # label x-axis
    plt.ylabel('Failure probability / fraction')  # label y-axis
    plt.title('Failure: theoretical vs empirical')  # add title
    plt.legend()  # show legend
    plt.grid(True)  # enable grid
    plt.tight_layout()  # improve layout
    plt.savefig(out)  # save figure to file
    print(f'Plot saved to {out}')  # notify where file is saved

####### EX3  # marker for exercise 3
def power_iter(P, s0, N=1000):  # forward power iteration: apply P repeatedly to a distribution
    n = P.shape[0]  # number of states
    if isinstance(s0, (list, tuple, np.ndarray)):  # if s0 is already a distribution
        mu = np.array(s0, dtype=float)  # use it directly
    else:
        mu = np.zeros(n, dtype=float)  # otherwise build a one-hot vector
        mu[s0] = 1.0  # put mass at s0

    for _ in range(N):  # iterate N times
        mu = mu @ P  # advance distribution
    return mu  # return resulting distribution


def stationary_eig(P, tol=1e-12):  # compute stationary distribution from eigenvector of P^T
    vals, vecs = np.linalg.eig(P.T)  # eigen-decomposition of transpose
    # find eigenvalues close to 1
    close = np.isclose(vals, 1.0, atol=tol)  # mask for eigenvalues ≈ 1
    if not np.any(close):
        raise RuntimeError("No eigenvalue 1 found")  # error if none
    # pick the eigenvector corresponding to the eigenvalue 1 with largest real part
    idxs = np.where(close)[0]  # indices with eigenvalue ≈ 1
    v = vecs[:, idxs[0]]  # choose the first matching eigenvector
    v = np.real_if_close(v)  # coerce to real if imaginary parts are negligible
    v = v.astype(float)  # ensure float dtype
    # normalize
    if v.sum() == 0:
        raise RuntimeError("Eigenvector sum is zero; cannot normalize")  # guard
    v = v / v.sum()  # normalize to sum to 1
    # enforce non-negativity (numerical)
    v[v < 0] = 0.0  # remove small negative noise
    v = v / v.sum()  # renormalize after clipping
    return v  # return stationary distribution

######## EX4  # marker for exercise 4
def mean_time_to_absorption(P, absorbing_indices):  # expected time to absorption for each state
    n = P.shape[0]  # total states
    absorbing = set(absorbing_indices)  # set of absorbing indices
    transient = [i for i in range(n) if i not in absorbing]  # transient state list
    if len(transient) == 0:
        return np.zeros(n)  # nothing to compute if no transient states
    Q = P[np.ix_(transient, transient)]  # submatrix for transient->transient transitions
    I = np.eye(Q.shape[0])  # identity of same size
    try:
        N = np.linalg.inv(I - Q)  # t=1+Qt <=> (I−Q)t=1 <=> t=(I−Q)^−1 fundamental matrix (I-Q)^{-1}
    except np.linalg.LinAlgError:
        # singular: return inf for transient states
        t = np.full(n, np.inf)  # default infinite times
        for a in absorbing:
            t[a] = 0.0  # absorbing states have 0 time to absorption
        return t  # return array
    ones = np.ones((Q.shape[0],))  # vector of ones
    t_transient = N.dot(ones)  # expected times for transient states
    t = np.full(n, 0.0)  # initialize full vector
    for idx, val in zip(transient, t_transient):  # map transient results back
        t[idx] = val  # assign expected time
    for a in absorbing:
        t[a] = 0.0  # absorbing states zero
    return t  # return expected times

# Same thing as the above Expected time = 1 step + expected future time
def mean_first_passage(P, target):  # mean first passage time to `target` from all states
    n = P.shape[0]  # number of states
    S = [i for i in range(n) if i != target]  # states excluding the target
    P_SS = P[np.ix_(S, S)]  # submatrix excluding target
    I = np.eye(P_SS.shape[0])  # identity
    ones = np.ones((P_SS.shape[0],))  # RHS ones
    h = np.full(n, np.inf)  # initialize with infinities
    try:
        x = np.linalg.solve(I - P_SS, ones)  # solve linear system for MFPTs
    except np.linalg.LinAlgError:
        return h  # singular -> return infinities
    for idx, val in zip(S, x):  # map solutions back into full vector
        h[idx] = val  # set MFPT
    h[target] = 0.0  # MFPT to target from target is zero
    return h  # return MFPT vector

####### EX5  # marker for exercise 5
# The graph demonstrates that the probability of failure peaks when weather is most unpredictable, specifically around a 0.5 weather stay probability, and is further exacerbated by low consumption stability (lower "cons stay" values). As weather patterns become more consistent and move toward a 0.9 stay probability, the risk of failure drops sharply across all scenarios, contradicting the idea that it converges to 1. Essentially, the system is most vulnerable when both weather and consumption are highly volatile, whereas high stability in either variable—but especially in weather—significantly improves system reliability.
def build_matrix_both_stay(weather_stay, consume_stay, scenario='B'):  # build P with variable stay probs
    """Build transition matrix varying both weather and consumption stay probabilities."""  # doc
    move_w = (1.0 - weather_stay) / 2.0  # split remaining mass for weather moves
    move_c = (1.0 - consume_stay) / 2.0  # split remaining mass for consumption moves
    P = np.zeros((N_TOTAL, N_TOTAL), dtype=float)  # init matrix
    for b in range(NB):  # iterate battery
        for w in range(NW):  # iterate weather
            for c in range(NC):  # iterate consumption
                i = idx(b, w, c)  # current index
                b_next = clip(b + w - c, 0, NB - 1)  # deterministic next battery
                # weather transitions
                w_probs = {}  # dict for weather outcome probs
                for dv, p in ((0, weather_stay), (-1, move_w), (1, move_w)):  # possible deltas
                    w2 = w + dv  # candidate weather
                    if 0 <= w2 <= NW - 1:  # check bounds
                        w_probs[w2] = w_probs.get(w2, 0.0) + p  # accumulate probability
                # consumption transitions with custom stay
                c_probs = {}  # dict for consumption outcome probs
                for dv, p in ((0, consume_stay), (-1, move_c), (1, move_c)):
                    c2 = c + dv  # candidate consumption
                    if 0 <= c2 <= NC - 1:  # check bounds
                        c_probs[c2] = c_probs.get(c2, 0.0) + p  # accumulate probability
                for w2, pw in w_probs.items():  # combine weather outcomes
                    for c2, pc in c_probs.items():  # combine consumption outcomes
                        j = idx(b_next, w2, c2)  # resulting state
                        P[i, j] += pw * pc  # add joint probability
    P[FAILURE_IDX, :] = 0.0  # failure row zeros
    P[FAILURE_IDX, FAILURE_IDX] = 1.0  # absorbing failure
    if scenario.upper() == 'B':  # apply scenario B adjustment if requested
        for i in range(N_OP):  # for each operational state
            prob_to_b0 = 0.0  # accumulate prob mass to b=0
            b0_inds = []  # list of target indices with b=0
            for j in range(N_OP):  # scan outgoing mass
                b_j, _, _ = inv_idx(j)  # get battery of target
                if b_j == 0:
                    prob_to_b0 += P[i, j]  # add to accumulator
                    b0_inds.append(j)  # record index
            if prob_to_b0 > 0.1 and b0_inds:  # if too much mass to b=0
                factor = (prob_to_b0 - 0.1) / prob_to_b0  # scaling factor
                for j in b0_inds:
                    P[i, j] *= factor  # reduce probabilities to b=0
                P[i, FAILURE_IDX] += 0.1  # move excess mass to failure
    row_sums = P.sum(axis=1)  # compute row sums
    for i in range(N_TOTAL):  # normalize rows to sum to 1
        if row_sums[i] > 0:
            P[i, :] /= row_sums[i]
    return P  # return matrix


def plot_failure_t100_vs_stays(ws_values=None, cs_values=None, s0=None, Ns=100,
                               out_png='failure_t100_vs_stays.png'):  # plot failure probability at t=Ns
    """Like the first panel of weather_sweep.png but at t=Ns, with one line per
    consumption stay probability. x-axis = weather stay prob."""  # docstring
    if ws_values is None:
        ws_values = np.linspace(0.05, 0.95, 19)  # default grid for weather stay
    if cs_values is None:
        cs_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # default consumption stay values
    if s0 is None:
        s0 = idx(2, 2, 1)  # default initial state

    plt.figure(figsize=(8, 5))  # create figure
    for cs in cs_values:  # loop over consumption stay values
        probs = []  # store failure probabilities for each weather stay
        for ws in ws_values:  # loop over weather stay values
            P = build_matrix_both_stay(ws, cs, scenario='B')  # build P for these parameters
            pi = np.zeros(P.shape[0]); pi[s0] = 1.0  # initial distribution at s0
            for _ in range(Ns):  # iterate Ns steps
                pi = pi @ P  # propagate distribution
            probs.append(pi[FAILURE_IDX])  # record probability of failure at time Ns
        plt.plot(ws_values, probs, '-o', label=f'cons stay={cs:.1f}')  # plot one curve per cs

    plt.xlabel('Weather stay probability')  # label x-axis
    plt.ylabel(f'Pr(failure at t={Ns})')  # label y-axis including Ns
    plt.title(f'Pr(failure) at t={Ns}: weather vs consumption volatility (Scenario B)')  # title
    plt.legend()  # show legend
    plt.grid(True)  # enable grid
    plt.tight_layout()  # tighten layout
    plt.savefig(out_png)  # save figure to file
    print(f'Plot saved to {out_png}')  # notify saved file

###### EX6:  # marker for exercise 6
#ETTF -> expected time to failure
def battery_level_comparison(b_values=None, scenario='C'):  # compare expected times by battery level
    if b_values is None:
        b_values = [2, 4]  # default battery values to compare
    P = build_matrix(scenario)  # build transition matrix for scenario
    t_abs = mean_time_to_absorption(P, [FAILURE_IDX])  # compute expected time to absorption (failure)

    print(f"{'State':<16} {'P(fail,t=1)':>12} {'ETTF':>8} {'mu_10':>8} {'mu_40':>8} {'mu_100':>8}  B_next(W=2,C=1)")  # header
    print('-' * 90)  # separator line
    summary = {b: [] for b in b_values}  # storage for ettf per battery value
    for b in b_values:
        for w in range(NW):
            for c in range(NC):
                s = idx(b, w, c)  # compute state index
                # direct one-step failure probability from the matrix row
                p_direct = P[s, FAILURE_IDX]  # probability to transition to failure in one step
                ettf = t_abs[s]  # expected time to failure for state s
                # transient failure probs
                pi = np.zeros(P.shape[0]); pi[s] = 1.0  # distribution starting at s
                vals = {}  # storage for selected time horizons
                for t in range(1, 101):  # compute up to 100 steps
                    pi = pi @ P  # advance distribution
                    if t in (10, 40, 100):
                        vals[t] = pi[FAILURE_IDX]  # record failure prob at selected times
                # b_next when W=2, C=1 as a representative case
                b_next_ex = clip(b + w - c, 0, NB - 1)  # example next battery level
                boundary = b_next_ex in (0, NB - 1)  # check if next battery is boundary
                print(f"(B={b},W={w},C={c})         {p_direct:>12.4f} {ettf:>8.2f} {vals[10]:>8.4f} {vals[40]:>8.4f} {vals[100]:>8.4f}  B'={b_next_ex} {'<BOUNDARY>' if boundary else ''}")  # print row
                summary[b].append(ettf)  # collect ettf
        print()  # blank line between battery sections

    # aggregated comparison
    print('=== Summary ===')  # summary header
    for b in b_values:
        ettfs = summary[b]  # list of ettfs for battery b
        print(f'B={b}  ETTF: min={min(ettfs):.2f}  max={max(ettfs):.2f}  mean={np.mean(ettfs):.2f}')  # print stats


if __name__ == "__main__":  # only run the following when script is executed directly
    print("Exercise 1")  # print progress marker
    # quick smoke test and save
    P_A = build_matrix('A')  # build matrix for scenario A
    P_B = build_matrix('B')  # build matrix for scenario B
    P_C = build_matrix('C')  # build matrix for scenario C
    print("P_A shape:", P_A.shape)  # show shape of P_A
    print("P_B shape:", P_B.shape)  # show shape of P_B
    print("P_C shape:", P_C.shape)  # show shape of P_C
    print("P_A row-sum min/max:", P_A.sum(axis=1).min(), P_A.sum(axis=1).max())  # check row sums
    print("P_B row-sum min/max:", P_B.sum(axis=1).min(), P_B.sum(axis=1).max())  # check row sums
    print("P_C row-sum min/max:", P_C.sum(axis=1).min(), P_C.sum(axis=1).max())  # check row sums
    print("P_A :", P_A)  # print matrix P_A
    print("P_B :", P_B)  # print matrix P_B
    print("P_C :", P_C)  # print matrix P_C

    print("Exercise 2")  # next exercise
    scenario = 'B'  # select scenario B for demonstration
    Ns = 40  # number of timesteps to analyze
    Nr = 5000  # number of Monte Carlo trajectories for simulation
    # choose initial state s0: e.g., B=2, W=2, C=1
    s0 = idx(2, 2, 1)  # compute linear index of example initial state

    P = build_matrix(scenario)  # build transition matrix for chosen scenario

    theory = theoretical_failure_probs(P, s0, Ns)  # compute theoretical failure probabilities
    empirical = simulate_trajectories(P, s0, Nr=Nr, Ns=Ns, seed=42)  # empirical Monte Carlo simulation

    # print a few values
    for t in [0, 1, 5, 10, 20, Ns]:  # selected timesteps
        print(f't={t}: theory={theory[t]:.6f}, empirical={empirical[t]:.6f}')  # show comparison

    plot_compare(theory, empirical)  # save a comparison plot
        
    print("Exercise 3")  # exercise 3 operations
    for N in [100, 1000, 5000, 20000]:  # different iteration counts
        muN = power_iter(P, s0, N=N)  # run power iteration
        print(f'power_iter N={N}: mu_N[75] = {muN[75]:.6f}')  # print probability for index 75

    # eigen-based stationary distribution
    pi = stationary_eig(P)  # compute stationary via eigenvector
    print(f'stationary_eig[75] = {pi[75]:.6f}')  # print value at index 75

    # L1 distance between large-N power-iter result and eigen stationary
    mu_large = power_iter(P, s0, N=20000)  # long-run power iteration
    l1 = np.linalg.norm(mu_large - pi, ord=1)  # compute L1 distance
    print(f'L1 distance (mu_20000, pi) = {l1:.6e}')  # print distance

    # multiplicity of eigenvalue 1 (diagnostic for uniqueness)
    mult = np.sum(np.isclose(np.linalg.eigvals(P.T), 1.0))  # count eigenvalues ≈ 1
    print(f'eigenvalue-1 multiplicity: {int(mult)}')  # print multiplicity
    
    print("Exercise 4")  # exercise 4 operations
    # Scenario C: compute expected time to failure (absorption)
    P_C = build_matrix('C')  # build matrix for scenario C

    for N in [100, 1000, 5000, 20000]:  # iterate different N
        muN = power_iter(P_C, s0, N=N)  # power iterate on P_C
        print(f'power_iter N={N}: mu_N[75] = {muN[75]:.6f}')  # print index 75

    # eigen-based stationary distribution
    pi = stationary_eig(P_C)  # stationary for P_C
    print(f'stationary_eig[75] = {pi[75]:.6f}')  # print value

    # L1 distance between large-N power-iter result and eigen stationary
    mu_large = power_iter(P_C, s0, N=20000)  # long-run power iter
    l1 = np.linalg.norm(mu_large - pi, ord=1)  # compute L1 norm
    print(f'L1 distance (mu_20000, pi) = {l1:.6e}')  # print

    # multiplicity of eigenvalue 1 (diagnostic for uniqueness)
    mult = np.sum(np.isclose(np.linalg.eigvals(P_C.T), 1.0))  # multiplicity
    print(f'eigenvalue-1 multiplicity: {int(mult)}')  # print multiplicity

    t_abs = mean_time_to_absorption(P_C, [FAILURE_IDX])  # expected time to absorption
    print(f'Expected time to failure from s0 (Scenario C): {t_abs[s0]:.4f} steps')  # print ettf

    # Mean first-passage time (s0 -> failure)
    h = mean_first_passage(P_C, FAILURE_IDX)  # compute MFPT to failure
    print(f'MFPT (s0 -> failure) = {h[s0]:.4f} steps')  # print mfpt

    # Example MFPT between two sample operational states
    s1 = idx(1, 1, 1)  # sample state s1
    s2 = idx(3, 3, 1)  # sample state s2
    h_s1_s2 = mean_first_passage(P_C, s2)  # MFPT to s2 from others
    print(f'MFPT (s1 -> s2) = {h_s1_s2[s1]:.4f} steps')  # print sample mfpt

    print('\nExercise 5')  # exercise 5 header
    plot_failure_t100_vs_stays()  # run plotting sweep

    print('\nExercise 6')  # exercise 6 header
    battery_level_comparison()  # run battery level comparison


