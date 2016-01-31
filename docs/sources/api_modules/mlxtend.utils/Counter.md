## Counter

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

