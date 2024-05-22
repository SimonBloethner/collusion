import numpy as np
import pandas as pd
import os
from scipy.stats import genpareto
from scipy.special import gamma, gammaincc
from datetime import datetime

start = datetime.now()

np.random.seed(1)
if str(os.path).find('Simon') > 0:
    data_location = '/Users/Simon/Documents/Projects/EWF VI/Research/PhD/Tails/data'
    out_location = '/Users/Simon/Documents/Projects/EWF VI/Research/PhD/Tails/data/simulations'
else:
    out_location = 'tradeTails/outData'
    data_location = 'tradeTails/Data'


df = pd.read_csv('{}/ITPDE_total.csv'.format(data_location), sep=',')

df = df.loc[np.logical_not(np.isnan(df.loc[:, 'distance'])), :]

beta_border = - 1	 # Trade cost constant.
int_border = np.where(df['exporter'] != df['importer'], 1, 0)
beta_rta = 0.5
beta_dist = 0.7
sig_ = 5
err_scale = 1
n_sim = 1000


def shadow_mean(xi, sigma, h, l):
    return (h - l) * np.exp((sigma / (h * xi)).astype(float)) * (sigma / (h * xi)) ** (1 / xi) * (gammaincc(((xi - 1) / xi).astype(float), (sigma / (h * xi)).astype(float)) * gamma(((xi - 1) / xi).astype(float))) + l


def phi(x, h, l):
    return l - h * np.log(((h - x) / (h - l)).astype(float))


def exports(t, E, p, P, sig=sig_):
    p_mat = np.tile(p, (t.shape[0], 1)).T
    P_mat = np.tile(P, (t.shape[0], 1))
    E_mat = np.tile(E, (t.shape[0], 1))

    return ((p_mat * t) / P_mat) ** (1 - sig) * E_mat


def price_index(t, p, sig=sig_):
    p_mat = np.tile(p, (t.shape[0], 1)).T

    return np.sum((p_mat * t) ** (1 - sig), axis=0) ** (1 / (1 - sig))


def price_prod(Y, t, E, P, sig=sig_):
    P_mat = np.tile(P, (t.shape[0], 1))
    E_mat = np.tile(E, (t.shape[0], 1))

    return (Y / np.sum((t / P_mat) ** (1 - sig) * E_mat, axis=1)) ** (1 / (1 - sig))


for sim in range(n_sim):
    cost_eq = beta_border * (int_border + np.random.normal(size=df.shape[0], scale=err_scale)) + beta_rta * (
                df['rta'] + np.random.normal(size=df.shape[0], scale=err_scale)) + beta_dist * (
                          np.log(df['distance']) + np.random.normal(size=df.shape[0], scale=err_scale))

    t = pd.DataFrame({'iso_x': df['exporter'],
                      'iso_i': df['importer'],
                      'year': df['year'],
                      'cost': np.exp(cost_eq) ** (1 / (1 - sig_)),
                      'rta': df['rta'], 'border': int_border,
                      'distance': df['distance']})

    strg = pd.DataFrame(columns=['exporter', 'importer', 'trade', 'year', 'iso_x', 'iso_i', 'cost', 'rta', 'border', 'distance'])

    for year in np.sort(df['year'].unique()):

        t_t = t.loc[t['year'] == year, :]

        t_t = t_t.loc[(t_t['iso_i'].isin(t_t['iso_x'].unique())) & (t_t['iso_x'].isin(t_t['iso_i'].unique()))]

        df_t = df.loc[t['year'] == year, :]

        df_t = df_t.loc[(df_t['importer'].isin(df_t['exporter'].unique())) & (df_t['exporter'].isin(df_t['importer'].unique()))]

        Y_t = df_t.groupby(['year', 'exporter'])['trade'].sum().reset_index()
        E_t = df_t.groupby(['year', 'importer'])['trade'].sum().reset_index()

        keep = Y_t.loc[Y_t['trade'] != 0, 'exporter'].values
        keep = list(set(keep) & set(E_t.loc[E_t['trade'] != 0, 'importer'].values))

        Y_t = Y_t[Y_t['exporter'].isin(keep)]
        E_t = E_t[E_t['importer'].isin(keep)]

        df_t = df_t[(df_t['importer'].isin(keep)) & (df_t['exporter'].isin(keep))]

        t_t = t_t[(t_t['iso_i'].isin(keep)) & (t_t['iso_x'].isin(keep))]

        dict_data = {'iso': pd.unique(t_t['iso_x']), 'number': range(len(pd.unique(t_t['iso_x'])))}
        dict_df = pd.DataFrame(dict_data)

        t_t = pd.merge(t_t, dict_df, left_on='iso_x', right_on='iso')
        t_t = t_t.drop(columns=['iso'])
        t_t = pd.merge(t_t, dict_df, left_on='iso_i', right_on='iso')
        t_t = t_t.drop(columns=['iso'])
        t_t.columns.values[(len(t_t.columns) - 2):] = ['nbr_ex', 'nbr_im']

        nrow = len(pd.unique(t_t['iso_x']))
        ncol = len(pd.unique(t_t['iso_i']))
        t_mat = np.empty((nrow, ncol))
        t_mat[:] = np.nan

        for edge in range(t_t.shape[0]):
            source = t_t.loc[edge, 'nbr_ex']
            target = t_t.loc[edge, 'nbr_im']
            weight = t_t.loc[edge, 'cost']
            t_mat[source, target] = weight
            t_mat[target, source] = weight  # Symmetric

        P = np.ones(len(np.unique(df_t['importer'])))
        p_small = np.ones(len(np.unique(df_t['importer'])))
        eps = np.full(len(np.unique(df_t['importer'])), 1e-6)
        diff_p = np.full(len(np.unique(df_t['importer'])), 10)

        while np.any(diff_p > eps):
            X = exports(t=t_mat, E=E_t.iloc[:, 2], p=p_small, P=P)
            P = price_index(t=t_mat, p=p_small)
            p_small_1 = price_prod(Y=Y_t.iloc[:, 2], t=t_mat, E=E_t.iloc[:, 2], P=P)
            p_small_1.iloc[np.where(dict_df.loc[:, 'iso'] == 'USA')[0]] = 1
            diff_p = np.abs(p_small - p_small_1)
            p_small = p_small_1

        num_nodes = X.shape[0]
        X_edge = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                weight = X[i, j]
                if weight > 0:
                    X_edge.append((i, j, weight))

        X_edge_df = pd.DataFrame(X_edge, columns=['exporter', 'importer', 'trade'])

        X_edge_df['year'] = year

        t_t = t_t.drop(columns=['year'])

        X_edge_df = pd.merge(X_edge_df, t_t, left_on=['exporter', 'importer'], right_on=['nbr_ex', 'nbr_im'])
        X_edge_df = X_edge_df.drop(columns=['nbr_ex', 'nbr_im'])

        strg = pd.concat([strg, X_edge_df])

    del X, X_edge, t_t, t, t_mat

    strg.drop(columns=['exporter', 'importer'], inplace=True)

    global_dict = pd.DataFrame({'iso': np.sort(np.unique(strg['iso_x'])), 'number': np.arange(len(np.unique(strg['iso_x'])))})

    strg = pd.merge(strg, global_dict, left_on='iso_x', right_on='iso')
    strg = strg.drop(columns=['iso'])
    strg = pd.merge(strg, global_dict, left_on='iso_i', right_on='iso')
    strg = strg.drop(columns=['iso'])

    cols = strg.columns.values[-2:]
    strg.rename(columns={k: v for k, v in zip(cols, ['exporter', 'importer'])}, inplace=True)

    strg = strg[strg['iso_x'] != strg['iso_i']]

    xmins = pd.DataFrame(data=np.nan, index=np.arange(len(np.unique(strg['year']))), columns=['year', 'estimate_xmin', 'lower_xmin'])

    n = 0
    for year in np.sort(np.unique(df['year'])):
        trade = strg.loc[strg['year'] == year, 'trade']

        top = trade[trade > 0]

        lower = np.quantile(trade, 0.90)
        top = top[top >= lower]

        xmins.loc[n, 'year'] = year
        # xmins.loc[n, 'estimate_xmin'] = estimate_xmin(m, xmins=top, xmax=max(trade))['xmin']
        xmins.loc[n, 'lower_xmin'] = lower

        n += 1

    year_max = strg.groupby('year')['trade'].sum().reset_index()

    strg.loc[:, 'trans'] = 0
    for year in sorted(strg['year'].unique()):
        year_sum = year_max.loc[year_max['year'] == year, 'trade'].values[0]
        x_min = xmins.loc[xmins['year'] == year, 'lower_xmin'].values[0]

        strg.loc[strg['year'] == year, 'trans'] = phi(strg.loc[strg['year'] == year, 'trade'], year_sum, x_min)

    ests = pd.DataFrame()

    for year in np.unique(strg['year']):
        thresh = xmins.loc[xmins['year'] == year, 'lower_xmin'].values[0]
        excess = strg.loc[strg['year'] == year, 'trans']
        excess = excess[excess > thresh]

        shape, loc, scale = genpareto.fit(excess)

        ests = pd.concat([ests, pd.DataFrame({'year': [year], 'xi': shape, 'beta': scale})], ignore_index=True)

    ests['year_max'] = year_max['trade']
    ests['lower'] = xmins['lower_xmin']

    ests['s_mean'] = shadow_mean(ests['xi'], ests['beta'], ests['year_max'], ests['lower'])

    strg = pd.merge(strg, ests[['year', 's_mean']], on='year')

    weights = pd.DataFrame({'year': sorted(strg['year'].unique()), 'body_mean': 0, 'tail_mean': 0})

    for year in strg['year'].unique():
        slice_data = strg[strg['year'] == year]

        theo_mean = ests.loc[ests['year'] == year, 's_mean'].values[0]

        body_mean = slice_data.loc[slice_data['trade'] < theo_mean, 'trade'].mean() * (
                    slice_data.loc[slice_data['trade'] < theo_mean].shape[0] / slice_data.shape[0])
        tail_mean = slice_data.loc[slice_data['trade'] > theo_mean, 'trade'].mean() * (
                    slice_data.loc[slice_data['trade'] > theo_mean].shape[0] / slice_data.shape[0])

        weights.loc[weights['year'] == year, 'body_mean'] = body_mean
        weights.loc[weights['year'] == year, 'tail_mean'] = tail_mean

    weights = pd.merge(weights, ests[['year', 's_mean']], on='year')

    weights['weight'] = (weights['s_mean'] - weights['body_mean']) / weights['tail_mean']

    strg = pd.merge(strg, weights[['year', 'weight']], on='year')

    strg.loc[strg['trade'] < strg['s_mean'], 'weight'] = 1

    strg = strg[['trade', 'year', 'iso_x', 'iso_i', 'rta', 'weight']]

    strg.to_csv('{}/{}.csv'.format(out_location, sim), index=False)

end = datetime.now()
print('Start time was: {}'.format(start))
print('End time was: {}'.format(end))
print('Execution time was {}'.format(datetime.now() - start))
