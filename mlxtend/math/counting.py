# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# math utilities to count the number of combinations and permutations

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
        
def num_combinations(n, r, with_replacement=False):
    """ 
    Function to calculate the number of possible combinations.
        
    Parameters
    ----------
    n : `int`
      Total number of items.
    r : `int`
      Number of elements of the target itemset.
    with_replacement : `bool`
      Allows repeated elements if True.
      
    Returns
    ----------
    comb : `int`
      Number of possible combinations.
        
    """
    if with_replacement:
        numerator = factorial(n + r - 1)
        denominator = factorial(r) * factorial(n-1)
    else:
        numerator = factorial(n)
        denominator = factorial(r) * factorial(n-r)
    comb = int(numerator/denominator)
    return comb
    
    
def num_permutations(n, r, with_replacement=False):
    """ 
    Function to calculate the number of possible permutations.
        
    Parameters
    ----------
    n : `int`
      Total number of items.
    r : `int`
      Number of elements of the target itemset.
    with_replacement : `bool`
      Allows repeated elements if True.
      
    Returns
    ----------
    permut : `int`
      Number of possible permutations.
        
    """
    if with_replacement:
        permut = n**r
    else:
        numerator = factorial(n)
        denominator = factorial(n-r)
        permut = int(numerator/denominator)
    return permut