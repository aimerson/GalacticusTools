#! /usr/bin/env python


class GalacticusError(Exception):
    """Base class for exceptions in this module."""
    pass


class ParseError(GalacticusError):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        super(ParseError, self).__init__(message)

        
