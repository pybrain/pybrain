#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-

""""This module is a place to hold functionality that _has_ to be outside of a 
test module but is required by it."""

# Used by test_tools_substitute
def otherfunc():
    print "I am the other func."