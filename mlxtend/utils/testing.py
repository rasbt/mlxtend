# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


def assert_raises(exception_type, message, func, *args, **kwargs):
    """Check that an exception is raised with a specific message

    Parameters
    ----------
    exception_type : exception
        The exception that should be raised
    message : str (default: None)
        The error message that should be raised. Ignored if False or None.
    func : callable
        The function that raises the exception
    *args : positional arguments to `func`.
    **kwargs : keyword arguments to `func`

    """
    try:
        func(*args, **kwargs)
    except exception_type as e:
        error_message = str(e)
        if message and message not in error_message:
            raise AssertionError(
                "Error message differs from the expected"
                " string: %r. Got error message: %r" % (message, error_message)
            )
    else:
        raise AssertionError("%s not raised." % exception_type.__name__)
