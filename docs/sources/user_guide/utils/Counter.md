# Counter

A simple progress counter to print the number of iterations and time elapsed in a for-loop execution.

> from mlxtend.utils import Counter

# Overview

The `Counter` class implements an object for displaying the number of iterations and time elapsed in a for-loop. Please note that the `Counter` was implemented for efficiency; thus, the `Counter` offers only very basic functionality in order to avoid relatively expensive evaluations (of if-else statements).

### References

- -

# Examples

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

# API


*Counter(stderr=False, start_newline=True)*

Class to display the progress of for-loop iterators.

**Parameters**

- `stderr` : bool (default: True)

    Prints output to sys.stderr if True; uses sys.stdout otherwise.

- `start_newline` : bool (default: True)

    Prepends a new line to the counter, which prevents overwriting counters
    if multiple counters are printed in succession.

**Attributes**

- `curr_iter` : int

    The current iteration.

- `start_time` : int

    The system's time in seconds when the Counter was initialized.

### Methods

<hr>

*update()*

Print current iteration and time elapsed.


