import sys
from src.main.python.lib.container import TraceContainer

from functools import partial

import numpy as np
import pandas as pd

from src.main.python.lib.math import contains_nan, get_hmm_model, correct_DA
import src.main.python.lib.math
import src.main.python.lib.plotting

from src.main.python.lib.container import TraceContainer

# path = 'src/test/resources/traces/Chan4_RNA01_flowin_selected_BCNL512_mol4of58FRET1to2_HaMMy.dat'

# key_hmmMode = 'E'
# trace = TraceContainer(filename=path, loaded_from_ascii=True,)

# traces = [trace]


# lDD, lDA, lAA, lE, llengths = [], [], [], [], []
# for trace in traces:
#     _, I_DD, I_DA, I_AA = correct_DA(trace.get_intensities())
#     lDD.append(I_DD[: trace.first_bleach])
#     lDA.append(I_DA[: trace.first_bleach])
#     lAA.append(I_AA[: trace.first_bleach])
#     lE.append(trace.fret[: trace.first_bleach])
#     llengths.append(len(I_DD[: trace.first_bleach]))

# E = np.array(lE)


# if key_hmmMode == "DA":
#     X = []
#     for ti in range(len(lDD)):
#         _x = np.column_stack((lDD[ti], lDA[ti], lAA[ti], lE[ti]))
#         X.append(_x)

#     if contains_nan([np.sum(aa) for aa in X[:][2]]):
#         X = [np.concatenate((_x[:, :2], _x[:, 3:]), axis=1) for _x in X]

#     X = np.array(X)
# else:
#     X = E.copy()


# E_flat = np.concatenate(E)

# best_mixture_model, params = src.main.python.lib.math.fit_gaussian_mixture(
#     E_flat,
#     min_n_components=1,
#     max_n_components=2, ##### change here to 2 and it will only fit two states I think? default is 6
#     strict_bic=False,
#     verbose=True,
# )
# n_components = best_mixture_model.n_components
# hmmModel = get_hmm_model(X, n_components=n_components)


# log_transmat = hmmModel.dense_transition_matrix()
# n_states = (hmmModel.node_count() - 2)  # minus virtual start and end state
# transmat = log_transmat[:n_states, :n_states]

# state_dict = {}
# for i, state in enumerate(hmmModel.states):
#     try:
#         if key_hmmMode == "DA":
#             state_dict[
#                 f"{state.name}".replace("s", "")
#             ] = state.distribution.parameters[0][-1].parameters
#         else:
#             state_dict[
#                 f"{state.name}".replace("s", "")
#             ] = state.distribution.parameters
#     except AttributeError:
#         continue
# means = np.array([v[0] for v in state_dict.values()])
# sigs = np.array([v[1] for v in state_dict.values()])

# print("Transition matrix:\n", np.round(transmat, 2))
# print("State means:\n", means)
# print("State sigmas:\n", sigs)


# for ti, trace in enumerate(traces):
#     _X = X[ti]
#     tf = pd.DataFrame()
#     tf["e_obs"] = trace.fret[: trace.first_bleach]
#     tf["state"] = np.array(hmmModel.predict(_X)).astype(int)
#     tf["e_pred_global"] = (
#         tf["state"]
#         .astype(str)
#         .replace(
#             {
#                 k: v[0]
#                 for (k, v) in zip(
#                     state_dict.keys(), state_dict.values()
#                 )
#             },
#             inplace=False,
#         )
#     )
#     tf["e_pred_local"] = tf.groupby(["state"], as_index=False)[
#         "e_obs"
#     ].transform("mean")

#     tf["time"] = tf["e_pred_local"].index + 1

#     trace.hmm_state = tf["state"].values
#     trace.hmm_local_fret = tf["e_pred_local"].values
#     trace.hmm_global_fret = tf["e_pred_global"].values
#     trace.hmm_idx = tf["time"].values

#     # trace.calculate_transitions()
# tf


# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.lineplot(data = tf, x = 'time', y = 'e_obs')
# sns.lineplot(data = tf, x = 'time', y = 'e_pred_local')
# plt.show()




# path = 'src/test/resources/traces/Chan4_RNA01_flowin_selected_BCNL512_mol4of58FRET1to2_HaMMy.dat'
# key_hmmMode = 'E'
# trace = TraceContainer(filename=path, loaded_from_ascii=True,)
# traces = [trace]
# data =  traces


path = 'src/test/resources/traces/Chan4_RNA01_flowin_selected_BCNL512_mol4of58FRET1to2_HaMMy.dat'
def import_data(path):
    trace = TraceContainer(filename=path, loaded_from_ascii=True,)
    return trace

traces = [trace]

def fit_HMM_NM(data):
    lDD, lDA, lAA, lE, llengths = [], [], [], [], []
    for trace in data:
        _, I_DD, I_DA, I_AA = correct_DA(trace.get_intensities())
        lDD.append(I_DD[: trace.first_bleach])
        lDA.append(I_DA[: trace.first_bleach])
        lAA.append(I_AA[: trace.first_bleach])
        lE.append(trace.fret[: trace.first_bleach])
        llengths.append(len(I_DD[: trace.first_bleach]))

    E = np.array(lE)


    if key_hmmMode == "DA":
        X = []
        for ti in range(len(lDD)):
            _x = np.column_stack((lDD[ti], lDA[ti], lAA[ti], lE[ti]))
            X.append(_x)

        if contains_nan([np.sum(aa) for aa in X[:][2]]):
            X = [np.concatenate((_x[:, :2], _x[:, 3:]), axis=1) for _x in X]

        X = np.array(X)
    else:
        X = E.copy()


    E_flat = np.concatenate(E)

    best_mixture_model, params = src.main.python.lib.math.fit_gaussian_mixture(
        E_flat,
        min_n_components=1,
        max_n_components=2, ##### change here to 2 and it will only fit two states I think? default is 6
        strict_bic=False,
        verbose=True,
    )
    n_components = best_mixture_model.n_components
    hmmModel = get_hmm_model(X, n_components=n_components)


    log_transmat = hmmModel.dense_transition_matrix()
    n_states = (hmmModel.node_count() - 2)  # minus virtual start and end state
    transmat = log_transmat[:n_states, :n_states]

    state_dict = {}
    for i, state in enumerate(hmmModel.states):
        try:
            if key_hmmMode == "DA":
                state_dict[
                    f"{state.name}".replace("s", "")
                ] = state.distribution.parameters[0][-1].parameters
            else:
                state_dict[
                    f"{state.name}".replace("s", "")
                ] = state.distribution.parameters
        except AttributeError:
            continue
    means = np.array([v[0] for v in state_dict.values()])
    sigs = np.array([v[1] for v in state_dict.values()])

    print("Transition matrix:\n", np.round(transmat, 2))
    print("State means:\n", means)
    print("State sigmas:\n", sigs)


    for ti, trace in enumerate(data):
        _X = X[ti]
        tf = pd.DataFrame()
        tf["e_obs"] = trace.fret[: trace.first_bleach]
        tf["state"] = np.array(hmmModel.predict(_X)).astype(int)
        tf["e_pred_global"] = (
            tf["state"]
            .astype(str)
            .replace(
                {
                    k: v[0]
                    for (k, v) in zip(
                        state_dict.keys(), state_dict.values()
                    )
                },
                inplace=False,
            )
        )
        tf["e_pred_local"] = tf.groupby(["state"], as_index=False)[
            "e_obs"
        ].transform("mean")

        tf["time"] = tf["e_pred_local"].index + 1

        trace.hmm_state = tf["state"].values
        trace.hmm_local_fret = tf["e_pred_local"].values
        trace.hmm_global_fret = tf["e_pred_global"].values
        trace.hmm_idx = tf["time"].values

        # trace.calculate_transitions()
    tf
    return tf

test = fit_HMM_NM(traces)


import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(data = test, x = 'time', y = 'e_obs')
sns.lineplot(data = test, x = 'time', y = 'e_pred_local')
plt.show()