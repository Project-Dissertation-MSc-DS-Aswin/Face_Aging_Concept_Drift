import sys
sys.path.append("../")

from copy import copy
import yaml

constants = yaml.load(open("../constants.yml", 'r').read())

class Args:
  
  def __new__(self, attrs):

    return type('ArgsDerived', (Args, ), attrs)

class Constants:
  
  def __new__(self, attrs=None):
  
    return type('ConstantsDerived', (Constants, ), constants)