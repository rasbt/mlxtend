mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>
# Combinations

> from mlxtend.math import num_combinations

Functions to calculate the number of combinations for creating subsequences of *r* elements out of a sequence with *n* elements.

## Example

In:

	from mlxtend.math import num_combinations

	c = num_combinations(n=20, r=8, with_replacement=False)
	print('Number of ways to combine 20 elements into 8 subelements: %d' % c)


Out:	

	Number of ways to combine 20 elements into 8 subelements: 125970

This is especially useful in combination with [`itertools`](https://docs.python.org/3/library/itertools.html), e.g., in order to estimate the progress via [`pyprind`](https://github.com/rasbt/pyprind).
    
   

![](./img/combinations_pyprind.png)

## Default Parameters

    def num_combinations(n, r, with_replacement=False):
        """ 
        Function to calculate the number of possible combinations.
        
        Parameters
        ----------
        n : `int`
          Total number of items.
      
        r : `int`
          Number of elements of the target itemset.
    
        with_replacement : `bool`, optional, (default=False)
          Allows repeated elements if True.
      
        Returns
        ----------
        comb : `int`
          Number of possible combinations.
        
        """
