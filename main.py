import heston
import pandas as pd
from optimize_param import method_of_moments
# import options  TODO: for plotting purposea

if __name__ == "__main__":
        
    r = 0.02 #rfr
    q = 0.0 #div yield
    s = 100.0 # spot price
    v = 0.04 # initial variance
    kappa = 1.0 # mean reversion speed
    theta = 0.04 # lr average variance
    sigma_v = 0.2 # volatility of variance
    rho = -0.7 #corr between asset returns and variance
    k = 100.0 # strike price
    tau = 1.0 # time to maturity (in years)

    # read data
    df = pd.read_csv("data.csv")

    print(df.head())

    # optimize 5 params
    r, theta, kappa, v, rho = method_of_moments(len(df)) 

    # run heston model
    heston.call(k, tau)