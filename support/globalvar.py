#!/usr/bin/python
# -*- coding: utf-8 -*-

def _init():
    global _global_dict
    _global_dict = {}

def set_value(name, value):
    if '_global_dict' not in globals():
        _init()
    _global_dict[name] = value

def get_value(name, defValue=None):
    if '_global_dict' not in globals():
        _init()
    try:
        return _global_dict[name]
    except KeyError:
        return defValue
