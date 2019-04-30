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
     [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 5.0, 0.0, 0.0],
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

  ("test relate left",
   SIMPLE_SCENE,
   r"relate(unique(\x.rock(x)),unique(\x.star(x)),left)",
   True),

  ("test relate left",
   SIMPLE_SCENE,
   r"relate(unique(\x.rock(x)),unique(\x.puddle(x)),left)",
   False),

  ("test relate right",
   SIMPLE_SCENE,
   r"relate(unique(\x.rock(x)),unique(\x.star(x)),right)",
   False),

  ("test relate right",
   SIMPLE_SCENE,
   r"relate(unique(\x.star(x)),unique(\x.rock(x)),right)",
   True),

  ("test relate right",
   SIMPLE_SCENE,
   r"relate(unique(\x.puddle(x)),unique(\x.rock(x)),right)",
   False),

  ("test relate_n 1",
   SIMPLE_SCENE,
   r"relate_n(unique(\x.star(x)),unique(\x.rock(x)),right,1)",
   True),

  ("test relate_n 2",
   SIMPLE_SCENE,
   r"relate_n(unique(\x.star(x)),unique(\x.spade(x)),up,2)",
   True),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.star(x)),up)",
   True),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.star(x)),right)",
   True),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.star(x)),down)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.star(x)),left)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.triangle(x)),up)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.triangle(x)),right)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.triangle(x)),down)",
   True),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.triangle(x)),left)",
   True),

  ("test is_edge",
   SIMPLE_SCENE,
   r"is_edge(unique(\x.circle(x)))",
   True),

  ("test is_edge",
   SIMPLE_SCENE,
   r"is_edge(unique(\x.star(x)))",
   True),

  ("test is_edge",
   SIMPLE_SCENE,
   r"is_edge(unique(\x.rock(x)))",
   True),

  ("test is_edge",
   SIMPLE_SCENE,
   r"is_edge(unique(\x.heart(x)))",
   False),

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
