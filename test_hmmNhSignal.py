'''
In this version we want to see how this approach is degrading when the signal is
becoming weaker and weaker

'''


import stan
import numpy as np

hmm = """
data {
    int<lower=0> T;             // num instances
    array[2] simplex[2] theta ;        // transit probs
    array[2] simplex[2] phi;          // emit probs
}
model {
}
generated quantities {
  int <lower=1,upper=2> z0;
  int <lower=1, upper=2> zt;
  array[T] int<lower=1, upper=2> w;  // words
  z0=1;
  for (t in 1:T)
   {
    zt = categorical_rng(theta[z0]);
    w[t] = categorical_rng(phi[zt]);
    z0=zt;
    }
}
  """

hmm_diag = """
data {
    int<lower=0> T;               // Number of time steps
    int<lower=1> K;               // Number of hidden states for each regime
    int<lower=1> V;               // Number of possible emissions
    array[T] int<lower=1, upper=2> y;   // Observed emissions

    // Transition and emission matrices for both regimes
    matrix<lower=0,upper=1>[K,K] A_regime1;      // Transition matrix for regime 1
    matrix<lower=0,upper=1>[K,V] B_regime1;      // Emission matrix for regime 1
    matrix<lower=0,upper=1>[K,K] A_regime2;      // Transition matrix for regime 2
    matrix<lower=0,upper=1>[K,V] B_regime2;      // Emission matrix for regime 2
    int<lower=1, upper=2> nh; // Health status
}

parameters {
    simplex[2] regime_probs;  // Probabilities of each regime
    // Hidden state probabilities for each regime

    array[T] simplex[K] hidden_states_prob_regime1;  // Hidden state probabilities for regime 1
    array[T] simplex[K] hidden_states_prob_regime2;  // Hidden state probabilities for regime 2
}

model {
    // Prior for regime probabilities
    regime_probs ~ dirichlet([1,1]);  // Equal priors for regimes

    // Likelihood for each regime
    hidden_states_prob_regime1[1] ~ dirichlet([1,1]);
    hidden_states_prob_regime2[1] ~ dirichlet([1,1]);

     for (t in 2:T) {
        // Likelihood for regime 1
        hidden_states_prob_regime1[t] ~ dirichlet(to_row_vector(hidden_states_prob_regime1[t - 1])*A_regime1); 
        y[t] ~ categorical(to_vector(to_row_vector(hidden_states_prob_regime1[t])*B_regime1));

        // Likelihood for regime 2
        hidden_states_prob_regime2[t] ~ dirichlet(to_row_vector(hidden_states_prob_regime2[t - 1])*A_regime2); 
        y[t] ~ categorical(to_vector(to_row_vector(hidden_states_prob_regime2[t])*B_regime2));

    }

    for (t in 1:T) {
    // Mixture model for regime selection based on observed emissions
    target += log_mix(regime_probs[1],
              categorical_lpmf(y[t] | to_vector(to_row_vector(hidden_states_prob_regime1[t])*B_regime1)),
              categorical_lpmf(y[t] | to_vector(to_row_vector(hidden_states_prob_regime2[t])*B_regime2)));
              }

}
"""

# Assuming nh is determined externally
nh_value = 1 # or 2, based on your criteria

duration = 500
good_regime = [[0.8, 0.2], [0.8, 0.2]]
bad_regime = [[0.7, 0.3], [0.5, 0.5]]
#sensor_emmission = [[1.0, 0.0], [0.0, 1.0]]  # perfect sensor
# sensor_emmission = [[0.6, 0.4], [0.4, 0.6]] # bad sensor

# Define signal strength: 1 (strong) to 0 (weak)
signal_strength = 0.8

# Adjust emission probabilities based on signal strength
sensor_emission_strong = [[1.0, 0.0], [0.0, 1.0]]  # Perfect sensor
sensor_emission_weak = [[0.6, 0.4], [0.5, 0.5]]  # Weaker sensor

# Linear interpolation between strong and weak signal based on signal_strength
sensor_emmission = [[(strong*signal_strength + weak*(1-signal_strength)) for strong, weak in zip(strong_sensor, weak_sensor)]
                   for strong_sensor, weak_sensor in zip(sensor_emission_strong, sensor_emission_weak)]

# Use sensor_emission in your hmm_data_good and hmm_data_bad dictionaries




if nh_value == 1:
    hmm_data_good = {
    "T": duration,
    "theta": good_regime,
    "phi": sensor_emmission,
    "nh": nh_value,
    }

    model = stan.build(hmm, data=hmm_data_good, random_seed=1)
    fit = model.fixed_param(num_chains=1, num_samples=1)
    # building data (nominal mode)
    wt = list(map(int, fit["w"]))

    hmm_diag_data = {
        "T": duration,
        "K": 2,
        "V": 2,
        "A_regime1": good_regime,  # more often in regime 1
        "A_regime2": bad_regime,
        "B_regime1": sensor_emmission,  # perfect sensor
        "B_regime2": sensor_emmission,
        "y": wt,  #usage of data
        "nh": nh_value,

    }

    model = stan.build(hmm_diag, data=hmm_diag_data, random_seed=1)

    fit = model.sample(num_chains=4, num_samples=100)
    nh_prob = fit["regime_probs"]
    print("With data from a good machine, nh=1")
    print(
        "probability of operation in nominal mode P(régime1)= ",
        np.mean(nh_prob[0]),
        "probability of operation in degraded mode P(régime2)= ",
        np.mean(nh_prob[1]),
    )

elif nh_value == 2:
    ##degraded mode
    hmm_data_bad = {
        "T": duration,
        "theta": bad_regime,
        "phi": sensor_emmission,
        "nh": nh_value,
    }
    model = stan.build(hmm, data=hmm_data_bad, random_seed=1)
    fit = model.fixed_param(num_chains=1, num_samples=1)
    # creation des données (mauvais régime)
    wt = list(map(int, fit["w"]))

    hmm_diag_data = {
        "T": duration,
        "K": 2,
        "V": 2,
        "A_regime1": good_regime,  # more often in regime 1
        "A_regime2": bad_regime,
        "B_regime1": sensor_emmission,  # perfect sensor
        "B_regime2": sensor_emmission,
        "y": wt,  # utilisation des données
        "nh": nh_value,
    }



    model = stan.build(hmm_diag, data=hmm_diag_data, random_seed=1)

    fit = model.sample(num_chains=4, num_samples=100)

    nh_prob = fit["regime_probs"]
    print("With data from a bad machine, nh=2")
    print(
        "probability of operation in nominal mode P(régime1)= ",
        np.mean(nh_prob[0]),
        "probability of operation in degraded mode P(régime2)= ",
        np.mean(nh_prob[1]),
    )
