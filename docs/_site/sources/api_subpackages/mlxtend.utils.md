mlxtend version: 0.15.0dev 
## Counter

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




## assert_raises

*assert_raises(exception_type, message, func, *args, **kwargs)*

Check that an exception is raised with a specific message

**Parameters**

- `exception_type` : exception

    The exception that should be raised

- `message` : str (default: None)

    The error message that should be raised. Ignored if False or None.

- `func` : callable

    The function that raises the exception

- `*args` : positional arguments to `func`.


- `**kwargs` : keyword arguments to `func`





## check_Xy

*check_Xy(X, y, y_int=True)*

None




## format_kwarg_dictionaries

*format_kwarg_dictionaries(default_kwargs=None, user_kwargs=None, protected_keys=None)*

Function to combine default and user specified kwargs dictionaries

**Parameters**

- `default_kwargs` : dict, optional

    Default kwargs (default is None).

- `user_kwargs` : dict, optional

    User specified kwargs (default is None).

- `protected_keys` : array_like, optional

    Sequence of keys to be removed from the returned dictionary
    (default is None).

**Returns**

- `formatted_kwargs` : dict

    Formatted kwargs dictionary.




