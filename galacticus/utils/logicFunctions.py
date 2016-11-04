#! /usr/bin/env python

def AND(*args):
    return [all(tuple) for tuple in zip(*args)]

def OR(*args):
    return [any(tuple) for tuple in zip(*args)]

