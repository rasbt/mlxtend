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


