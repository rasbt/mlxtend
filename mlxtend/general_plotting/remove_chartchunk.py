# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# matplotlib utilities for removing chartchunk

def remove_borders(axes, left=False, bottom=False, right=True, top=True):
    """ 
        A function to remove chartchunk from matplotlib plots, such as axes
        spines, ticks, and labels.
        
        Keyword arguments:
            axes: An iterable containing plt.gca() or plt.subplot() objects, e.g. [plt.gca()].
            left, bottom, right, top: Boolean to specify which plot axes to hide.
            
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
