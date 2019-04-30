from functools import partial

from frozendict import frozendict
from nose.tools import *
import numpy as np

from pyccg.logic import Expression
from pyccg.model import Model

from puddleworldOntology import ontology, process_scene


SIMPLE_SCENE = np.array(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 2.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 5.0, 8.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])



CASES = [
  ("test basic object predicate",
   SIMPLE_SCENE,
   r"unique(\x.spade(x))",
   frozendict(row=2, col=9, type="spade")),

  ("test pick",
   SIMPLE_SCENE,
   r"move(unique(\x.spade(x)))",
   (2, 9)),

  ("test relate down",
   SIMPLE_SCENE,
   r"relate(unique(\x.spade(x)),unique(\x.puddle(x)),down)",
   True),

  ("test relate down",
   SIMPLE_SCENE,
   r"relate(unique(\x.puddle(x)),unique(\x.spade(x)),down)",
   False),

  ("test relate up",
   SIMPLE_SCENE,
   r"relate(unique(\x.spade(x)),unique(\x.puddle(x)),up)",
   False),

  ("test relate up",
   SIMPLE_SCENE,
   r"relate(unique(\x.puddle(x)),unique(\x.spade(x)),up)",
   True),

]


def _test_case(scene, expression, expected, msg=None):
  from pprint import pprint
  print("Objects:")
  pprint(scene['objects'])

  model = Model(scene, ontology)
  expr = Expression.fromstring(expression)
  value = model.evaluate(expr)
  print(expr)
  print("Expected:", expected)
  print("Observed:", value)

  eq_(value, expected, msg)


def test_fixed_cases():
  for msg, scene, expression, expected in CASES:
    scene = process_scene([scene])

    f = partial(_test_case)
    f.description = "test case: %s -> %s" % (expression, expected)
    yield f, scene, expression, expected, msg
