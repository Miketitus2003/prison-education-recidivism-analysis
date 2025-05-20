import numpy as np
import pandas as pd

def simulate_dataset(seed=42, n=1000):
    """
    Simulates a dataset based on post-release employment, violent offense history,
    and time served to predict recidivism outcomes.

    Parameters:
        seed (int): Random seed for reproducibility
        n (int): Number of simulated individuals

    Returns:
        pd.DataFrame: Simulated dataset
    """
    np.random.seed(seed)
    employed = np.random.binomial(1, 0.5, n)
    violent_offense = np.random.binomial(1, 0.4, n)
    time_served = np.round(np.random.uniform(0.5, 10, n), 1)

    linear_pred = 0.9 * violent_offense - 0.98 * employed + 0.05 * time_served
    prob_recidivism = 1 / (1 + np.exp(-linear_pred))
    recidivism = np.random.binomial(1, prob_recidivism)

    df = pd.DataFrame({
        "Employed": employed,
        "Violent_Offense": violent_offense,
        "Time_Served": time_served,
        "Recidivism": recidivism
    })

    return df
