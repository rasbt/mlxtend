# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Algorithm for plotting sequential feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt

def plot_sequential_feature_selection(metric_dict,
                                      kind='std_dev',
                                      color='blue',
                                      bcolor='steelblue',
                                      marker='o',
                                      alpha=0.2,
                                      ylabel='Performance',
                                      confidence_interval=0.95):
    """Plot feature selection results.

    Parameters
    ----------
    metric_dict : mlxtend.SequentialFeatureSelector.get_metric_dict() object
    kind : str (default: "std_dev")
        The kind of error bar or confidence interval in
        {'std_dev', 'std_err', 'ci', None}.
    color : str (default: "blue")
        Color of the lineplot (accepts any matplotlib color name)
    bcolor : str (default: "steelblue").
        Color of the error bars / confidence intervals
        (accepts any matplotlib color name).
    marker : str (default: "o")
        Marker of the line plot
        (accepts any matplotlib marker name).
    alpha : float in [0, 1] (default: 0.2)
        Transparency of the error bars / confidence intervals.
    ylabel : str (default: "Performance")
        Y-axis label.
    confidence_interval : float (default: 0.95)
        Confidence level if `kind='ci'`.

    Returns
    ----------
    fig : matplotlib.pyplot.figure() object

    """

    allowed = {'std_dev', 'std_err', 'ci', None}
    if kind not in allowed:
        raise AttributeError('kind not in %s' % allowed)

    fig = plt.figure()
    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]['avg_score'] for k in k_feat]

    if kind:
        upper, lower = [], []
        if kind == 'ci':
            kind = 'ci_bound'

        for k in k_feat:
            upper.append(metric_dict[k]['avg_score'] +
                         metric_dict[k][kind])
            lower.append(metric_dict[k]['avg_score'] -
                         metric_dict[k][kind])

        plt.fill_between(k_feat,
                         upper,
                         lower,
                         alpha=alpha,
                         color=bcolor,
                         lw=1)

        if kind == 'ci_bound':
            kind = 'Confidence Interval (%d%%)' % (confidence_interval * 100)

    plt.plot(k_feat, avg, color=color, marker=marker)
    plt.ylabel(ylabel)
    plt.xlabel('Number of Features')
    feature_min = len(metric_dict[k_feat[0]]['feature_idx'])
    feature_max = len(metric_dict[k_feat[-1]]['feature_idx'])
    plt.xticks(range(feature_min, feature_max + 1),
               range(feature_min, feature_max + 1))
    return fig
