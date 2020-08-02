"""
Utilities and data generation functions for plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression


iv_columns = ['instrument', 'treatment', 'confounder', 'outcome']

def generate_iv_data(n_samples, treat_effect, confound_effect, I_T, C_T, seed=42, exclusion_effect=0):
    """
    Generates IV data from a multivariate Gaussian.
    
    n_samples (int): the number samples to generate
    treat_effect (float): the true effect (coefficient) of the treatment on outcome
    confound_effect (float): the effect of the confounder (coefficient) on the outcome
    I_T (float): instrument strength, the covariance between the instrument and treatment
    C_T (float): confound strength, the covariance between the confound and treatment
    seed (int): optionally set random seed, for reproducibility
    exclusion_effect (float): optionally set the effect of the instrument on the outcome, violating the exclusion restriction
    """
    
    idx_dict = {
        'I': 0,
        'T': 1,
        'C': 2,
        'O': 3
    }

    # vars:             I    T    C
    covar = np.array([[1.0, I_T, 0.0], # I
                      [I_T, 1.0, C_T], # T
                      [0.0, C_T, 1.0]])# C
    covar += np.eye(3,3)
    # vars:  I  T  C
    means = [0, 0, 0]

    # generate some data
    np.random.seed(seed)
    data = np.random.multivariate_normal(mean=means, cov=covar, size=n_samples)

    O = (confound_effect*data[:, idx_dict['C']]) \
        + (treat_effect*(data[:, idx_dict['T']])) \
        + (exclusion_effect*data[:, idx_dict['I']]) \
        + np.random.normal(0,1,size=n_samples)

    data = np.concatenate([data,O.reshape(-1, 1)], axis=1)

    data_df = pd.DataFrame(data, columns=iv_columns)
    
    return data_df


def plot_iv_scatter(x, y, data_df, ax):
    """
    Plots IV data on scatter plot.
    """
    scatter_kws = {
        'alpha': 0.3
    }

    line_kws = {
        'color': 'red'
    }
    reg = LinearRegression(fit_intercept=True)
    reg.fit(data_df[[x]], data_df[y])
    
    corr= np.corrcoef(data_df[x], data_df[y])[0,1]
    sns.regplot(x=x, y=y, data=data_df, 
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                label="reg coef: {:.2f}\ncorr: {:.2f}".format(reg.coef_[0], corr),
                ax=ax)
    ax.legend()


def plot_iv_dist(n_trials, n_samples, treat_effect, confound_effect, I_T, C_T, ax, exclusion_effect=0, kde=True):
    reg_estimates = []
    iv_estimates = []
    for t in range(n_trials):
        df = generate_iv_data(n_samples, treat_effect, confound_effect, I_T, C_T, seed=t, exclusion_effect=exclusion_effect)

        reg = LinearRegression(fit_intercept=True)
        reg.fit(df[['treatment']], df['outcome'])
        reg_estimates.append(reg.coef_)

        stage1 = LinearRegression(fit_intercept=True)
        stage1.fit(df[['instrument']], df['treatment'])
        t_hat = stage1.predict(df[['instrument']])
        stage2 = LinearRegression(fit_intercept=True)
        stage2.fit(t_hat.reshape(-1,1), df['outcome'])
        iv_estimates.append(stage2.coef_)

    ax = plt.gca()
    ax.axvline(x=treat_effect, color='black', ls='--', label="true treatment effect")

    sns.distplot(reg_estimates, ax=ax, label="reg estimate", kde=kde)
    sns.distplot(iv_estimates, ax=ax, label="iv estimate", kde=kde)
    ax.set_ylabel("density")
    ax.set_xlabel("estimated treatment effect")
    ax.legend()


max_time = 10
cutoff = max_time / 2

def generate_did_data(n_samples, treat_time_effect, control_time_effect, control_offset, treat_effect, seed):
    """
    Generates simple diff-in-diff data.
    
    Args:
        n_samples (int): number of samples to generate
        treat_time_effect (float): 
        control_time_effect (float): 
        control_offset (float): 
        seed (int):
    """
    np.random.seed(seed)
    treat_indicator = np.random.choice(2, size=n_samples)
    time = np.random.uniform(low=0, high=max_time, size=n_samples)

    outcome = (treat_time_effect * treat_indicator * time) + \
              (control_time_effect * (1-treat_indicator) * time) + \
              (control_offset * (1-treat_indicator)) + \
              (treat_effect * treat_indicator * (time > cutoff).astype(int)) + \
              np.random.normal(0,1, size=n_samples)


    did_df = pd.DataFrame()
    did_df['time'] = time
    did_df['treat_indicator'] = treat_indicator
    did_df['pre_post_indicator'] = (time > cutoff).astype(int)
    did_df['outcome'] = outcome
    
    return did_df


def plot_did_scatter(did_df, ax):
    """
    Plots the diff in diff scatter plot.
    
    """
    sns.scatterplot(x='time', y='outcome', hue='treat_indicator', 
                    data=did_df, alpha=0.3, legend=False, ax=ax)
    ax.axvline(x=cutoff, label="X (treat) cutoff", color='black', ls='--')

    control_pre = did_df.loc[(did_df['time'] < cutoff) & (did_df['treat_indicator'] == 0)]
    control_post = did_df.loc[(did_df['time'] >= cutoff) & (did_df['treat_indicator'] == 0)]

    treat_pre = did_df.loc[(did_df['time'] < cutoff) & (did_df['treat_indicator'] == 1)]
    treat_post = did_df.loc[(did_df['time'] >= cutoff) & (did_df['treat_indicator'] == 1)]
    
    x_pre = np.linspace(0,5).reshape(-1,1)
    x_post = np.linspace(5,10).reshape(-1,1)
    
    ctl_pre_reg = LinearRegression().fit(control_pre[['time']], control_pre['outcome']).predict(x_pre)
    ctl_post_reg = LinearRegression().fit(control_post[['time']], control_post['outcome']).predict(x_post)

    trt_pre_reg = LinearRegression().fit(treat_pre[['time']], treat_pre['outcome']).predict(x_pre)
    trt_post_reg = LinearRegression().fit(treat_post[['time']], treat_post['outcome']).predict(x_post)

    
    ax.plot(x_pre, ctl_pre_reg, ls="--", label="control Y", color="C0")
    ax.plot(x_post, ctl_post_reg, ls="--", color="C0")
    
    ax.plot(x_pre, trt_pre_reg, ls="--", label="treat Y", color="orange")
    ax.plot(x_post, trt_post_reg, ls="--", color="orange")
    ax.legend()


def plot_did_dist(n_trials, n_samples, treat_time_effect, control_time_effect, control_offset, treat_effect, ax):
    diff_estimates = []
    did_estimates = []
    for t in range(n_trials):
        did_df = generate_did_data(n_samples, treat_time_effect, control_time_effect, control_offset, treat_effect, seed=t)

        control_pre = np.mean(did_df.loc[(did_df['time'] < cutoff) & (did_df['treat_indicator'] == 0), 'outcome'])
        control_post = np.mean(did_df.loc[(did_df['time'] >= cutoff) & (did_df['treat_indicator'] == 0), 'outcome'])

        treat_pre = np.mean(did_df.loc[(did_df['time'] < cutoff) & (did_df['treat_indicator'] == 1), 'outcome'])
        treat_post = np.mean(did_df.loc[(did_df['time'] >= cutoff) & (did_df['treat_indicator'] == 1), 'outcome'])

        diff_estimates.append((treat_post - treat_pre))
        did_estimates.append((treat_post - treat_pre) - (control_post - control_pre))
        
    ax.axvline(x=treat_effect, color='black', ls='--', label="true treatment effect")

    sns.distplot(diff_estimates, ax=ax, label="single difference estimate")
    sns.distplot(did_estimates, ax=ax, label="diff in diff estimate")
    ax.set_ylabel("density")
    ax.set_xlabel("estimated treatment effect")
    ax.legend()


def generate_rdd_data(n_samples, treat_effect, confound_effect, C_R, seed=42, nonlinear=False):
    """
    Generates sharp RDD data.
    
    n_samples (int): the number samples to generate
    treat_effect (float): the true effect (coefficient) of the treatment on outcome
    confound_effect (float): the effect of the confounder (coefficient) on the outcome
    running_effect (float): the effect of the running variable on the outcome, in addition to the treatment effect
    C_R (float): confound strength, the covariance between the confound and running var
    nonlinear (bool): whether or not to generate a nonlinear/linear 
    """
    # vars:             R    C 
    covar = np.array([[1.0, C_R],  # R
                      [C_R, 1.0]]) # C
    covar += np.eye(2,2)
    # vars:  R, C
    means = [0, 0]

    # generate some data
    np.random.seed(seed)
    data = np.random.multivariate_normal(mean=means, cov=covar, size=n_samples)    
    
    running = data[:,0]
    confound = data[:,1]
    
    running = np.random.uniform(low=-10, high=10, size=n_samples)
    confound = running + np.random.normal(0,1, size=n_samples)
    sel_samples = int(C_R*n_samples)
    #confound[np.random.choice(n_samples, size=sel_samples, replace=False)] = np.random.normal(0,1, size=sel_samples)
    
#     print(np.corrcoef(running, confound)[0,1])
    
    treat = (running > 0).astype(int)
    
    #outcome = (confound_effect*confound) + (treat_effect*treat) + (running_effect * running) + np.random.normal(0,1,size=n_samples)
    outcome = (confound_effect*confound) + (treat_effect*treat) + np.random.normal(0,2,size=n_samples)

    if nonlinear:
        outcome = (confound_effect/10 * confound)**3 + (treat_effect*treat) + np.random.normal(0,1,size=n_samples)
    
    rdd_df = pd.DataFrame()
    rdd_df['running'] = running
    rdd_df['confound'] = confound
    rdd_df['treat'] = treat
    rdd_df['outcome'] = outcome
    
    return rdd_df


def plot_rdd_scatter(x,y, rdd_df, bandwidth, treat_effect, ax):
    
    sns.scatterplot(x=x, y=y, data=rdd_df, alpha=0.5, ax=ax)
    if (x == "running") and (y == "outcome"):
        ax.axvline(x=0, color="black", ls="--", label = "treatment cutoff")
        
        x_pre = np.linspace(-1 * bandwidth, 0).reshape(-1,1)
        x_post = np.linspace(0, bandwidth).reshape(-1,1)
        
        left_cutoff = rdd_df[(rdd_df['running'] < 0) & (rdd_df['running'] > -1*bandwidth)]
        right_cutoff = rdd_df[(rdd_df['running'] > 0) & (rdd_df['running'] < bandwidth)]
        
        left_reg = LinearRegression().fit(left_cutoff[['running']], left_cutoff['outcome']).predict(x_pre)
        right_reg = LinearRegression().fit(right_cutoff[['running']], right_cutoff['outcome']).predict(x_post)

        ax.plot(x_pre, left_reg, color="red", label="local regressions")
        ax.plot(x_post, right_reg, color="red",)
        padding = 50
        right_rect = patches.Rectangle((0,rdd_df['outcome'].min()-padding),bandwidth,(rdd_df['outcome'].max() + padding)*2,
                                 linewidth=0,
                                 color='r',
                                 fill=True,
                                 alpha=0.25)
        
        left_rect = patches.Rectangle((-1*bandwidth,rdd_df['outcome'].min()-padding),bandwidth,(rdd_df['outcome'].max() + padding)*2,
                                 linewidth=0,
                                 color='r',
                                 fill=True,
                                 alpha=0.25)
        ax.add_patch(left_rect)
        ax.add_patch(right_rect)
        ax.legend()


def plot_rdd_dist(n_trials, n_samples, treat_effect, confound_effect, C_R, bandwidth, ax, nonlinear=False):
    reg_estimates = []
    rdd_estimates = []
    for t in range(n_trials):
        rdd_df = generate_rdd_data(n_samples, treat_effect, confound_effect, C_R, seed=t, nonlinear=nonlinear)

        reg = LinearRegression().fit(rdd_df[['treat']], rdd_df['outcome'])
        reg_estimates.append(reg.coef_)

        left_cutoff = rdd_df[(rdd_df['running'] > (-1*bandwidth)) & (rdd_df['running'] < 0)]
        right_cutoff = rdd_df[(rdd_df['running'] > 0) & (rdd_df['running'] < bandwidth)]
        
        left_reg = LinearRegression().fit(left_cutoff[['running']], left_cutoff['outcome']).intercept_
        right_reg = LinearRegression().fit(right_cutoff[['running']], right_cutoff['outcome']).intercept_
        
        rdd_estimates.append(right_reg - left_reg)

    ax = plt.gca()
    ax.axvline(x=treat_effect, color='black', ls='--', label="true treatment effect")

    sns.distplot(reg_estimates, ax=ax, label="reg estimate")
    sns.distplot(rdd_estimates, ax=ax, label="rdd estimate")
    ax.set_ylabel("density")
    ax.set_xlabel("estimated treatment effect")
    ax.legend()