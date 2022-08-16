import sys
import os
sys.path.append("../src")

from copy import copy
import yaml

class Args:
  
  def __new__(self, attrs):

    return type('ArgsDerived', (Args, ), attrs)

class Constants:
  
  def __new__(self, attrs=None):
  
    return type('ConstantsDerived', (Constants, ), constants)