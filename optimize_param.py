'''
Discretization:
- Forward Euler (this is the least accurate but easiest)

Note: don't need Cuckoo Search. Our paper presents two approaches toward parameter estimation
- Moment in moments (fits non-convex spaces)
- MLE (Maximum Likelihood Estimation)
'''
import numpy as np
from utils import *
from scipy.optimize import fsolve, least_squares # fitting paraeters

### DISCRETIZATION SCHEME ###

# def next_S(S, V, r):
#     S_t1 = S + r * S + np.sqrt(V) * S * brownian(0, 1)
#     S = S_t1
#     V = next()

# def next_V(V, k, theta)

def gen_Q_V(V_0, r, theta, k, sigma, end_t):
    '''
    For now, forward Euler discretiation scheme
    - Higher-order discretization is far more complicated (Milstein method). Conventional methods like RK4 won't work

    We have some of the 5 Heston model parameters that we need to optimize for here, but this isn't a problem
    - Iteratively run this and improve our estimates

    V_0: initial volatility
    k: kappa, rate of mean reversion on long-term price variance
    rho: corr between 2 Wiener processes
        - this is a hyperparameter in overall Heston model, but just run separate correlation for this discretization
    theta: LT rate of volatility mean reversion
    end_t: last timestep

    Note: two parameters that we are optimizating for are in here so we need to recalculate every instance: kappa/k, sigma
    V_0
    - Can set to theta
    - Or, calculate as variance of returns (log) or variance of underlying asset price.
    - Need to clarify which option we want 

    Note: Q is percent change in S (asset returns): \frac{S_{t+1}}{S_t}
    - This means we don't directly need S for this

    Z represents our Wiener processes/Brownian motion
    '''
    Q_vals = [0]
    V_vals = [V_0]

    V = V_0

    Z_1 = [brownian(0, 1) for _ in range(end_t)]
    Z_2 = [brownian(0, 1) for _ in range(end_t)]
    # rho = np.corrcoef(Z_1, Z_2)[0, 1]
    rho = estimate_rho(Z_1, Z_2) # normally Z_1 with vol, but here paper wants both Wiener Processes

    # Q_{t+1} = 1 + r + \sqrt{V_t} \left(\rho Z_t + \sqrt{1-\rho^2} Z_2 \right)
    # V_{t+1} = V_t + k(\theta - V_t) + \sigma np.sqrt{V_t} Z_1
    for i in range(end_t):
        Q = 1 + r + np.sqrt(V) * (rho * Z_1[i] + np.sqrt(1- rho**2) * Z_2[i])
        V = V + k * (theta-V) + sigma * np.sqrt(V) * Z_1[i]

        Q_vals.append(Q)
        V_vals.append(V)
    
    return (Q_vals, V_vals, rho)

def convert_S_to_Q(S):
    '''
    Utility
    - Convert stock/asset prices to pct change
    '''
    for i in range(1, len(S)):
        S[i] = S[i]/ S[i-1]
    
    S[0] = 0

    return S

def gen_V(data, log_transform = True):
    '''
    This is variance based on log-transformed S i.e. % change in S

    Not sure if this is supposed to be variance of S price (as paper says) or log-transformed returns
    - We basically log-transform S in our calibration, but the paper textually says variance of asset price
    '''
    if log_transform:
        return np.mean([(np.log10(data[i]) - np.log10(data[i-1])) ** 2 for i in range(1, len(data))])
    
    return np.std(data) ** 2

### PARAMETER OPTIMIZATION SCHEME ###

### METHOD OF MOMENTS ###

def method_of_moments(end_t, V_data = None):
    '''
    We want 5 moments of Q_{t+1} in terms of five parameters r,k,theta, sigma, rho

    nth moment: \mu_n = \int_{-\infty}^\infty (x-c)^n f(x) dx
    https://en.wikipedia.org/wiki/Moment_(mathematics)
    https://web.stanford.edu/class/archive/cs/cs109/cs109.1218/files/student_drive/7.3.pdf

    Simpler alternative to MLE
    - doesn't estimate rho, but paper asserts this isn't a big deal.

    Iteratively fit variables based on each moment. Each moment allows us to solve for one more parameter (excluding rho)

    @end_t: specifies length of log-transformed S (Q)
    @V_data (optional): source S data for calculating V_0 (variance/squared stdev of log-transformed S)
    '''
    # Initial guess for (r, theta, k, sigma)
    init_guess = [0.02, 0.04, 100.00, 0.2]
    # if we believe our market is in long-run equilibrium, we can set v_0 = theta
    # Need someone to further research v_0. This should be fine, just wondering if there's benefits to otherwise calculating v_0
    V = init_guess[1]

    def heston_moments(params):
        """Defines the system of equations for method of moments estimation."""
        r, theta, k, sigma = params

        # recalculate QV and emp_moments
        Q = gen_Q_V(V, init_guess[2], init_guess[1], init_guess[0], init_guess[3], end_t)

        emp_moments = np.array([np.mean(Q**i) for i in range(1, 6)])
        mu1, mu2, _, mu4, mu5 = emp_moments

        # mu_1
        eq1 = mu1 - (1 + r)  # Solve for r
        
        # mu_2
        eq2 = mu2 - ((r + 1)**2 + theta)  # Solve for theta

        # mu_4
        f1 = 1 / (k * (k - 2))
        f2 = np.sum([k**2 * r**4, 4*k**2 * r**3, 6*k**2 * r**2 * theta, -2*k*r**4, 6*k**2 * r**2, 12*k**2 * r*theta,
                     3*k**2*theta**2, -8*k*r**3, -12*k*r**2*theta, 4*k**2*r, 6*k**2*theta, -12*k*r**2, -24*k*r*theta,
                     -6*k*theta**2, -3*sigma**2 * theta, k**2,-8*k*r,-12*k*theta, -2*k])
        eq3 = mu4 - f1 * f2  # Solve for k, sigma

        # mu_5
        f3 = np.sum([k**2 * r**5, 5*k**2 * r**4, 10*k**2*r**3*theta, -2*k*r**5, 10*k**2*r**3, 30*k**2*r**2*theta,
                     15*k**2*r*theta**2, -10*k*r**4, -20*k*r**3*theta, 10*k**2*r**2, 30*k**2 * r*theta, 15*k**2*theta**2,
                     -20*k*r**3,-60*k*r**2*theta, -30*k*r*theta**2, -15*r*sigma**2*theta, 5*k**2 *r, 10*k**2*theta, -20*k*r**2,
                     -60*k*r*theta,-30*k*theta**2, -15*sigma**2*theta, k**2, -10*k*r, -20*k*theta, -2*k])
        eq4 = mu5 - (1 / (k * (k - 2))) * f3  # Refit sigma

        # min least squares all 4 eq's at same time
        return np.array([eq1, eq2, eq3, eq4])

    # Solve using least squares (more stable than fsolve)
    result = least_squares(heston_moments, init_guess)
    # r, theta, k, sigma = result.x
    return tuple(result.x)

def estimate_rho(W, V):
    return np.corrcoef(W,V)[0,1]