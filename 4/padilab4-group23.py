import math
import numpy as np
import matplotlib.pyplot as plt

def golf_env_step(env, x, a):

    if env == 0:
        xgoal = 200 # the position of the hole
        wind = -10 #bias for the ball's movement
    elif env == 1:
        xgoal = 200
        wind = 10
    elif env == 2:
        xgoal = 400
        wind = -20
    elif env == 3:
        xgoal = 400
        wind = +10

    if a == 0:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 10 + 100) + wind
    elif a == 1:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 30 + 150) + wind
    elif a == 2:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 5 + 50) + wind/2
    elif a == 3:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 15 + 75) + wind/2
    elif a == 4:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 1) + min((xgoal-x),10)
    elif a == 5:
        x = x + np.sign(xgoal-x) * (np.random.normal() * 4 + 2 * min((xgoal-x),20)) + wind/4

    x = np.double(x) # convert to double for more precision
    if x < 0:
        x = 0
    elif x > 450:
        x = 450

    if np.abs(x-xgoal) < 2: # if the ball is within 2 units of the hole, it's considered a success
        c = 0 # success
        x = 0
    else:
        c = 1 # failure

    return x, c

ENV = 2 # The golf course to consider.
Na = 6

def x_to_features(x):
  """Convert an array of positions x (shape (N,)) into feature matrix (2 x N)."""
  x = np.asarray(x, dtype=float).flatten() # ensure x is a 1D array of floats
  f1 = x / 450.0 # normalize position to [0, 1]
  ones = np.ones_like(f1) # bias term (always 1)
  return np.vstack([f1, ones]) # stack features into a 2 x N matrix

x_test_1 = np.array([311.9])
f_test_1 = x_to_features(x_test_1)
print(f_test_1)
print(f_test_1.shape)

x_test_2 = np.array([0.0, 86.6, 148.1, 231.5, 311.9])
f_test_2 = x_to_features(x_test_2)
print(f_test_2)
print(f_test_2.shape)

def updateCoef(T, theta, gamma = 0.9, Na = 6, solver = 'lstsq', sgd_lr=1e-3, sgd_iters=1000):

    next_theta = np.copy(theta)

    # If no transitions provided, return current parameters unchanged
    if T is None or len(T) == 0:
        return next_theta

    T = np.array(T)
    xs = T[:, 0] # current states
    acts = T[:, 1].astype(int) # actions (ensure they are integers)
    costs = T[:, 2] # costs
    xns = T[:, 3] # next states

    for a in range(Na):
        idx = np.where(acts == a)[0]
        if idx.size == 0:
            continue

        x_a = xs[idx] # current states for action a
        xn_a = xns[idx] # next states for action a
        c_a = costs[idx] # costs for action a

        # Features for current states (N_a x 2)
        F = x_to_features(x_a).T

        # Features for next states and compute min_a' Q(x',a')
        Fnext = x_to_features(xn_a).T
        qnext = Fnext @ theta        # shape (N_a, Na)
        min_qnext = np.min(qnext, axis=1)

        # Regression targets: y = c + gamma * min_a' Q(x',a')
        y = c_a + gamma * min_qnext

        if solver == 'lstsq':
        # Solve least squares F theta_a = y for theta_a (2,)
            theta_a, *_ = np.linalg.lstsq(F, y, rcond=None)
            next_theta[:, a] = theta_a.flatten()
        elif solver == 'sgd':
            w = next_theta[:, a].copy()
            for _ in range(sgd_iters):
                pred = F @ w
                grad = (F.T @ (pred - y)) / max(1, len(y))
                w -= sgd_lr * grad
            next_theta[:, a] = w.flatten()
        else:
            raise ValueError('Unknown solver: ' + str(solver))

    return next_theta

np.random.seed(1640)
theta = np.ones([2,Na]) / Na
T = np.array([[  0.        ,   0.        ,   1.        ,  77.97101407],
       [ 77.97101407,   0.        ,   1.        , 147.42896569],
       [147.42896569,   0.        ,   1.        , 216.11126557],
       [216.11126557,   0.        ,   1.        , 308.66565669],
       [308.66565669,   0.        ,   1.        , 373.28914596]])
updated_theta = updateCoef(T, theta, Na, solver='lstsq')
updated_theta_sgd = updateCoef(T, theta, Na, solver='sgd')
print(updated_theta)
print(updated_theta.shape)

print(updated_theta_sgd)
print(updated_theta_sgd.shape)

#alternative to match with reinforce
def FittedQI(env, gamma=0.99, Niter=2000, MaxSteps=30, theta = np.ones([2,Na]) / Na):
    egreedy = 0.1
    T = []
    x = 0
    LogFQI = []

    _, nA = theta.shape

    for i in range(Niter):
        # 1. Generate a full trajectory
        x = 0
        C = 0
        for _ in range(MaxSteps):

            if np.random.rand() < egreedy:
                a = np.random.randint(0, nA)
            else:
                a = np.argmin(x_to_features(np.array([x])).T@theta)

            xn,c = golf_env_step(ENV, x, a)
            C = C + c
            T.append([x,a,c,xn])
            x = xn
            if c == 0:
                break

        LogFQI.append(C)
        #
        if len(T)%100==0:
            theta = updateCoef(np.array(T), theta, gamma = gamma, Na = nA, solver = 'lstsq')
            #T = []
    return theta, LogFQI, T


#Verify if it works
theta, LogFQI, T = FittedQI(ENV, gamma=0.99, Niter=2000, MaxSteps=30)
plt.plot(LogFQI)
plt.xlabel('episodes')
plt.ylabel('length of episode') # because the cost is 1 per step until success, the total cost is equal to the length of the episode until the ball goes in the hole.

#This code allows visualizing the policy

xs = np.linspace(0.0, 450.0, 100) # 100 test positions from 0 to 450
ftoplot = np.concatenate([x_to_features(np.array([x])) for x in xs], axis=1)
print(ftoplot.shape)
plt.plot(xs, ftoplot.T@theta) # this will plot the Q-values for each action as a function of the position x. The optimal action at each position is the one with the lowest Q-value (since we are minimizing cost).
plt.legend(['0','1','2','3','4','5'])
# plt.plot(np.min(ftoplot.T@theta,axis=1))


#alternative to match with reinforce
def FittedQI2(env, gamma=0.99, Niter=2000, MaxSteps=30, theta = np.ones([2,Na]) / Na):
    egreedy = 0.1
    T = []
    x = 0
    LogFQI = []

    _, nA = theta.shape

    for i in range(Niter):
        # 1. Generate a full trajectory
        x = 0
        C = 0
        for _ in range(MaxSteps):

            if np.random.rand() < egreedy:
                a = np.random.randint(0, nA)
            else:
                a = np.argmin(x_to_features(np.array([x])).T@theta)

            xn,c = golf_env_step(ENV, x, a)
            C = C + c
            T.append([x,a,c,xn])
            x = xn
            if c == 0:
                break

        LogFQI2.append(C)
        #
        if len(T)%100==0:
            theta = updateCoef(np.array(T), theta, gamma = gamma, Na = nA, solver = 'sgd')
            #T = []
    return theta, LogFQI2, T


#Verify if it works
theta, LogFQI2, T = FittedQI2(ENV, gamma=0.99, Niter=2000, MaxSteps=30)
plt.plot(LogFQI2)
plt.xlabel('episodes')
plt.ylabel('length of episode') # because the cost is 1 per step until success, the total cost is equal to the length of the episode until the ball goes in the hole.

#This code allows visualizing the policy

xs = np.linspace(0.0, 450.0, 100) # 100 test positions from 0 to 450
ftoplot = np.concatenate([x_to_features(np.array([x])) for x in xs], axis=1)
print(ftoplot.shape)
plt.plot(xs, ftoplot.T@theta) # this will plot the Q-values for each action as a function of the position x. The optimal action at each position is the one with the lowest Q-value (since we are minimizing cost).
plt.legend(['0','1','2','3','4','5'])
# plt.plot(np.min(ftoplot.T@theta,axis=1))

def softmaxProb(f, theta):
    """Compute softmax policy probabilities for features f and parameters theta."""
    f = np.asarray(f, dtype=float).flatten() # ensure f is a 1D array of floats
    prefs = theta.T @ f  # shape (Na,) - action preferences
    exps = np.exp(prefs)
    probs = exps / np.sum(exps)
    return probs

def reinforceUpdate(states, actions, costs, theta, lr = 0.005, gamma = 0.99):
    """Perform REINFORCE updates on theta given an episode."""
    T = len(states)
    theta = np.array(theta, dtype=float)
    for t in range(T):
        # Compute return G_t = sum_{k=t}^{T-1} gamma^{k-t} * costs[k]
        G = 0.0
        for k in range(t, T):
            G += (gamma**(k-t)) * costs[k]

        f = np.asarray(states[t], dtype=float).flatten() #
        probs = softmaxProb(f, theta)

        # Update parameters for all actions: theta[:,a] 
        # One-hot encoding of action
        one_hot = np.zeros_like(probs)
        one_hot[actions[t]] = 1

        # Gradient ∇θ log π(a|s)
        grad_log_pi = np.outer(f, (one_hot - probs))

        # Update
        theta -= lr * (gamma**t) * G * grad_log_pi # para a question 6 -> lr * G * grad_log_pi

    return theta

theta = np.array([[-2.7, -1.06, -1.2, -0.72,  6.31, -0.59],
       [-0.0047,  4.47, -1.477, 0.31, -1.47, -1.827]])
f = x_to_features(np.array([0])).flatten()
probs = softmaxProb(f, theta)
print(probs)

new_theta = reinforceUpdate( [f], [3], [1], theta)
print(new_theta)

def reinforce(env, Na, lr=0.005, gamma=0.99, Niter=2000, MaxSteps=300):
    # Initialize parameters (2 features x Na actions)
    theta = np.zeros((2, Na))
    Log = []

    for i in range(Niter):
        # 1. Generate a full trajectory
        states, actions, costs = [], [], []
        x = 0
        for _ in range(MaxSteps):
            f = x_to_features(np.array([x])).flatten()

            # Compute softmax probabilities
            probs = softmaxProb(f, theta)

            # Sample action from policy
            a = np.random.choice(Na, p=probs)

            # Step the environment
            xn, c = golf_env_step(env, x, a)

            states.append(f)
            actions.append(a)
            costs.append(c)

            x = xn
            if c == 0:
                break
        Log.append( np.sum(costs) )

        # 2. Update policy parameters
        theta = reinforceUpdate(states, actions, costs, theta, lr = lr, gamma = gamma)


    return theta, Log

# Train with REINFORCE
print("Training REINFORCE...")
theta_reinforce, LogReinforce = reinforce(env=ENV, Na=6, Niter=2000, lr=0.01)

plt.plot(LogReinforce)
plt.xlabel('episodes')
plt.ylabel('length of episode')

plt.plot(LogReinforce)
plt.plot(LogFQI)
plt.legend(['Reinforce', 'FQI'])
plt.xlabel('episodes')
plt.ylabel('length of episode')
