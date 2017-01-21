# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for removing chart junk from matplotlib plots
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


def remove_borders(axes, left=False, bottom=False, right=True, top=True):
    """Remove chart junk from matplotlib plots.

    Parameters
    ----------
    axes : iterable
        An iterable containing plt.gca()
        or plt.subplot() objects, e.g. [plt.gca()].
    left : bool (default: `False`)
        Hide left axis spine if True.
    bottom : bool (default: `False`)
        Hide bottom axis spine if True.
    right : bool (default: `True`)
        Hide right axis spine if True.
    top : bool (default: `True`)
        Hide top axis spine if True.

    """
    for ax in axes:
        ax.spines["top"].set_visible(not top)
        ax.spines["right"].set_visible(not right)
        ax.spines["bottom"].set_visible(not bottom)
        ax.spines["left"].set_visible(not left)
        if bottom:
            ax.tick_params(bottom="off", labelbottom="off")
        if top:
            ax.tick_params(top="off")
        if left:
            ax.tick_params(left="off", labelleft="off")
        if right:
            ax.tick_params(right="off")
