# Counter

A simple progress counter to print the number of iterations and time elapsed in a for-loop execution.

> from mlxtend.utils import Counter

## Overview

The `Counter` class implements an object for displaying the number of iterations and time elapsed in a for-loop. Please note that the `Counter` was implemented for efficiency; thus, the `Counter` offers only very basic functionality in order to avoid relatively expensive evaluations (of if-else statements).

### References

- -

## Example 1 - Counting the iterations in a for-loop


```python
from mlxtend.utils import Counter
```


```python
import time

cnt = Counter()
for i in range(20):
    # do some computation
    time.sleep(0.1)
    cnt.update()
```

    
    20 iter | 2 sec

Note that the first number displays the current iteration, and the second number shows the time elapsed after initializing the `Counter`.

## API


*Counter(stderr=False, start_newline=True, precision=0, name=None)*

Class to display the progress of for-loop iterators.

**Parameters**

- `stderr` : bool (default: True)

    Prints output to sys.stderr if True; uses sys.stdout otherwise.

- `start_newline` : bool (default: True)

    Prepends a new line to the counter, which prevents overwriting counters
    if multiple counters are printed in succession.
    precision: int (default: 0)
    Sets the number of decimal places when displaying the time elapsed in
    seconds.

- `name` : string (default: None)

    Prepends the specified name before the counter to allow distinguishing
    between multiple counters.

**Attributes**

- `curr_iter` : int

    The current iteration.

- `start_time` : float

    The system's time in seconds when the Counter was initialized.

- `end_time` : float

    The system's time in seconds when the Counter was last updated.

**Examples**


    >>> cnt = Counter()
    >>> for i in range(20):
    ...     # do some computation
    ...     time.sleep(0.1)
    ...     cnt.update()
    20 iter | 2 sec
    >>> print('The counter was initialized.'
    ' %d seconds ago.' % (time.time() - cnt.start_time))
    The counter was initialized 2 seconds ago
    >>> print('The counter was last updated'
    ' %d seconds ago.' % (time.time() - cnt.end_time))
    The counter was last updated 0 seconds ago.

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/utils/Counter/](http://rasbt.github.io/mlxtend/user_guide/utils/Counter/)

### Methods

<hr>

*update()*

Print current iteration and time elapsed.


