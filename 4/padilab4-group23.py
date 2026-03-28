import math  # math utilities (not heavily used but available)
import numpy as np  # numerical arrays and math operations
import matplotlib.pyplot as plt  # plotting library for visualizations

def golf_env_step(env, x, a):
    """Step the golf environment: apply action `a` from position `x` on course `env`.

    Returns the new position `x` and cost `c` (0 on success, 1 otherwise).
    """

    if env == 0:
        xgoal = 200  # target hole position for environment 0
        wind = -10  # horizontal wind bias for env 0
    elif env == 1:
        xgoal = 200  # same hole position but opposite wind
        wind = 10  # positive wind bias for env 1
    elif env == 2:
        xgoal = 400  # different hole position for env 2
        wind = -20  # stronger negative wind in env 2
    elif env == 3:
        xgoal = 400  # env 3 shares hole with env 2
        wind = +10  # positive wind for env 3

    # Action effects: each action samples a noisy displacement biased toward the hole
    if a == 0:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 10 + 100) + wind  # strong shot
    elif a == 1:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 30 + 150) + wind  # very strong, more variance
    elif a == 2:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 5 + 50) + wind / 2  # medium shot
    elif a == 3:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 15 + 75) + wind / 2  # medium-strong shot
    elif a == 4:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 1) + min((xgoal - x), 10)  # short, controlled putt-like move
    elif a == 5:
        x = x + np.sign(xgoal - x) * (np.random.normal() * 4 + 2 * min((xgoal - x), 20)) + wind / 4  # adaptive medium shot

    x = np.double(x)  # ensure floating precision
    if x < 0:
        x = 0  # lower bound on position
    elif x > 450:
        x = 450  # upper bound on position

    # success condition: within 2 units of hole
    if np.abs(x - xgoal) < 2:  # ball is considered in the hole
        c = 0  # success has zero cost
        x = 0  # reset position for convenience
    else:
        c = 1  # step cost is 1 until success

    return x, c  # return new position and immediate cost

ENV = 2  # chosen environment index for examples
Na = 6  # number of discrete actions available

def x_to_features(x):
        """Convert positions `x` (shape (N,) or scalar) to feature matrix (2 x N).

        Features: [normalized position, bias].
        """
        x = np.asarray(x, dtype=float).flatten()  # ensure 1D float array
        f1 = x / 450.0  # normalized position in [0,1]
        ones = np.ones_like(f1)  # bias term
        return np.vstack([f1, ones])  # return 2 x N feature matrix

x_test_1 = np.array([311.9])  # single example position
f_test_1 = x_to_features(x_test_1)  # features for single example
print(f_test_1)  # debug print of features
print(f_test_1.shape)  # shape should be (2, 1)

x_test_2 = np.array([0.0, 86.6, 148.1, 231.5, 311.9])  # multiple test positions
f_test_2 = x_to_features(x_test_2)  # features for multiple positions
print(f_test_2)  # debug print
print(f_test_2.shape)  # shape should be (2, 5)

def updateCoef(T, theta, gamma=0.9, Na=6, solver='lstsq', sgd_lr=1e-3, sgd_iters=1000):
    """Fit linear Q-function parameters `theta` from dataset `T`.

    T is an array of transitions [x, a, c, xn]. For each action we solve
    a regression for the action-specific linear parameters.
    """

    next_theta = np.copy(theta)  # copy to avoid in-place overwrite

    # Return early if no data
    if T is None or len(T) == 0:
        return next_theta

    T = np.array(T)
    xs = T[:, 0]  # current states (positions)
    acts = T[:, 1].astype(int)  # actions as integers
    costs = T[:, 2]  # observed costs
    xns = T[:, 3]  # next states (positions)

    for a in range(Na):
        idx = np.where(acts == a)[0]  # indices where action == a
        if idx.size == 0:
            continue  # no samples for this action

        x_a = xs[idx]  # positions for action a
        xn_a = xns[idx]  # next positions for action a
        c_a = costs[idx]  # costs for action a

        # Design matrix for current states: shape (N_a, 2)
        F = x_to_features(x_a).T

        # Features for next states and compute min_a' Q(x',a') using current theta
        Fnext = x_to_features(xn_a).T
        qnext = Fnext @ theta  # Q-values for all actions at next states, shape (N_a, Na)
        min_qnext = np.min(qnext, axis=1)  # min over actions (greedy target)

        # Regression targets: y = c + gamma * min_a' Q(x',a')
        y = c_a + gamma * min_qnext

        if solver == 'lstsq':
            # Solve least squares F theta_a = y for action-specific params theta_a
            theta_a, *_ = np.linalg.lstsq(F, y, rcond=None)
            next_theta[:, a] = theta_a.flatten()
        elif solver == 'sgd':
            # Simple SGD for the same regression
            w = next_theta[:, a].copy()
            for _ in range(sgd_iters):
                pred = F @ w
                grad = (F.T @ (pred - y)) / max(1, len(y))
                w -= sgd_lr * grad
            next_theta[:, a] = w.flatten()
        else:
            raise ValueError('Unknown solver: ' + str(solver))

    return next_theta

np.random.seed(1640)  # reproducible randomness for examples
theta = np.ones([2, Na]) / Na  # initial parameter guess (2 x Na)
T = np.array([
    [0.0, 0.0, 1.0, 77.97101407],
    [77.97101407, 0.0, 1.0, 147.42896569],
    [147.42896569, 0.0, 1.0, 216.11126557],
    [216.11126557, 0.0, 1.0, 308.66565669],
    [308.66565669, 0.0, 1.0, 373.28914596],
])  # small example dataset of transitions
updated_theta = updateCoef(T, theta, Na=Na, solver='lstsq')  # fit with least squares
updated_theta_sgd = updateCoef(T, theta, Na=Na, solver='sgd')  # fit with SGD
print(updated_theta)  # show fitted parameters
print(updated_theta.shape)  # shape should be (2, Na)

print(updated_theta_sgd)  # show SGD result
print(updated_theta_sgd.shape)

#alternative to match with reinforce
def FittedQI(env, gamma=0.99, Niter=2000, MaxSteps=30, theta=np.ones([2, Na]) / Na):
    """Fitted Q-Iteration: collect transitions, periodically fit linear Q approximators."""
    egreedy = 0.1  # exploration probability when sampling
    T = []  # dataset of transitions
    x = 0  # start position
    LogFQI = []  # record cumulative costs per generated episode

    _, nA = theta.shape  # number of actions

    for i in range(Niter):
        # generate one episode / trajectory
        x = 0
        C = 0
        for _ in range(MaxSteps):
            if np.random.rand() < egreedy:
                a = np.random.randint(0, nA)  # random action
            else:
                a = np.argmin(x_to_features(np.array([x])).T @ theta)  # greedy action (min cost)

            xn, c = golf_env_step(env, x, a)  # step environment
            C = C + c  # accumulate cost
            T.append([x, a, c, xn])  # store transition
            x = xn
            if c == 0:
                break  # reached hole

        LogFQI.append(C)  # log cumulative cost for this episode

        if len(T) % 100 == 0:
            theta = updateCoef(np.array(T), theta, gamma=gamma, Na=nA, solver='lstsq')  # refit
            # optionally clear T to use only recent samples
    return theta, LogFQI, T


#Verify if it works
theta, LogFQI, T = FittedQI(ENV, gamma=0.99, Niter=2000, MaxSteps=30)  # run FQI example
plt.plot(LogFQI)  # plot episode lengths over iterations
plt.xlabel('episodes')
plt.ylabel('length of episode')  # cost==episode length until success

#This code allows visualizing the policy

xs = np.linspace(0.0, 450.0, 100)  # grid of test positions
ftoplot = np.concatenate([x_to_features(np.array([x])) for x in xs], axis=1)  # features for each x
print(ftoplot.shape)  # should be (2, 100)
plt.plot(xs, ftoplot.T @ theta)  # plot Q-values (per action) vs position
plt.legend(['0', '1', '2', '3', '4', '5'])  # action labels
# optional: plt.plot(np.min(ftoplot.T @ theta, axis=1)) to plot min-Q


#alternative to match with reinforce
def FittedQI2(env, gamma=0.99, Niter=2000, MaxSteps=30, theta=np.ones([2, Na]) / Na):
    """Same as FittedQI but uses SGD solver for fitting."""
    egreedy = 0.1
    T = []
    x = 0
    LogFQI = []

    _, nA = theta.shape

    for i in range(Niter):
        x = 0
        C = 0
        for _ in range(MaxSteps):
            if np.random.rand() < egreedy:
                a = np.random.randint(0, nA)
            else:
                a = np.argmin(x_to_features(np.array([x])).T @ theta)

            xn, c = golf_env_step(env, x, a)
            C = C + c
            T.append([x, a, c, xn])
            x = xn
            if c == 0:
                break

        LogFQI.append(C)

        if len(T) % 100 == 0:
            theta = updateCoef(np.array(T), theta, gamma=gamma, Na=nA, solver='sgd')
    return theta, LogFQI, T


#Verify if it works
theta, LogFQI2, T = FittedQI2(ENV, gamma=0.99, Niter=2000, MaxSteps=30)  # run SGD-based FQI
plt.plot(LogFQI2)  # plot its episode lengths
plt.xlabel('episodes')
plt.ylabel('length of episode')  # cost == episode length until success

# Visualize learned Q-values across positions
xs = np.linspace(0.0, 450.0, 100)  # test grid
ftoplot = np.concatenate([x_to_features(np.array([x])) for x in xs], axis=1)
print(ftoplot.shape)
plt.plot(xs, ftoplot.T @ theta)  # Q-values per action vs position
plt.legend(['0', '1', '2', '3', '4', '5'])

def softmaxProb(f, theta):
    """Return softmax action probabilities for feature vector `f` and params `theta`."""
    f = np.asarray(f, dtype=float).flatten()  # ensure 1D feature vector
    prefs = theta.T @ f  # action preference scores (Na,)
    exps = np.exp(prefs - np.max(prefs))  # numerically stable softmax
    probs = exps / np.sum(exps)
    return probs

def reinforceUpdate(states, actions, costs, theta, lr=0.005, gamma=0.99):
    """Apply REINFORCE policy-gradient updates to `theta` using one episode."""
    T = len(states)
    theta = np.array(theta, dtype=float)
    for t in range(T):
        # compute discounted return from time t
        G = 0.0
        for k in range(t, T):
            G += (gamma ** (k - t)) * costs[k]

        f = np.asarray(states[t], dtype=float).flatten()
        probs = softmaxProb(f, theta)

        # one-hot for taken action
        one_hot = np.zeros_like(probs)
        one_hot[actions[t]] = 1

        # gradient of log-probability: outer product f × (one_hot - probs)
        grad_log_pi = np.outer(f, (one_hot - probs))

        # gradient ascent on expected return (we subtract because costs are positive; keep sign convention)
        theta -= lr * (gamma ** t) * G * grad_log_pi

    return theta

theta = np.array([[-2.7, -1.06, -1.2, -0.72, 6.31, -0.59],
            [-0.0047, 4.47, -1.477, 0.31, -1.47, -1.827]])  # example parameters
f = x_to_features(np.array([0])).flatten()  # feature at position 0
probs = softmaxProb(f, theta)  # example softmax probabilities
print(probs)

new_theta = reinforceUpdate([f], [3], [1], theta)  # tiny update example
print(new_theta)

def reinforce(env, Na, lr=0.005, gamma=0.99, Niter=2000, MaxSteps=300):
    """REINFORCE policy-gradient algorithm using softmax policy parameterization."""
    theta = np.zeros((2, Na))  # initialize policy parameters
    Log = []

    for i in range(Niter):
        states, actions, costs = [], [], []
        x = 0
        for _ in range(MaxSteps):
            f = x_to_features(np.array([x])).flatten()

            # compute action probabilities and sample
            probs = softmaxProb(f, theta)
            a = np.random.choice(Na, p=probs)

            xn, c = golf_env_step(env, x, a)

            states.append(f)
            actions.append(a)
            costs.append(c)

            x = xn
            if c == 0:
                break
        Log.append(np.sum(costs))  # record episode cost / length

        # update policy parameters with REINFORCE
        theta = reinforceUpdate(states, actions, costs, theta, lr=lr, gamma=gamma)

    return theta, Log

# Train with REINFORCE
print("Training REINFORCE...")
theta_reinforce, LogReinforce = reinforce(env=ENV, Na=6, Niter=2000, lr=0.01)  # run REINFORCE

plt.plot(LogReinforce)
plt.xlabel('episodes')
plt.ylabel('length of episode')

plt.plot(LogReinforce)
plt.plot(LogFQI)
plt.legend(['Reinforce', 'FQI'])
plt.xlabel('episodes')
plt.ylabel('length of episode')
