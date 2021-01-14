class StaleValueException(Exception):
    """Used to avoid pullling values that have already been used."""

    def __init__(self):
        Exception.__init__(self, "Trying to use stale value.")