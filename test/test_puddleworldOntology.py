#### If need to add adjacent libraries.
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../ec/")
sys.path.insert(0, "../pyccg")
sys.path.insert(0, "../pyccg/nltk")
####

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
   r"unique(\x.diamond(x))",
   frozendict(row=2, col=9, type="diamond")),

  ("test pick",
   SIMPLE_SCENE,
   r"move(unique(\x.diamond(x)))",
   (2, 9)),

  ("test relate down",
   SIMPLE_SCENE,
   r"relate(unique(\x.diamond(x)),unique(\x.star(x)),down)",
   True),

  ("test relate down",
   SIMPLE_SCENE,
   r"relate(unique(\x.star(x)),unique(\x.diamond(x)),down)",
   False),

  ("test relate up",
   SIMPLE_SCENE,
   r"relate(unique(\x.diamond(x)),unique(\x.star(x)),up)",
   False),

  ("test relate up",
   SIMPLE_SCENE,
   r"relate(unique(\x.star(x)),unique(\x.diamond(x)),up)",
   True),

  ("test relate left",
   SIMPLE_SCENE,
   r"relate(unique(\x.rock(x)),unique(\x.star(x)),left)",
   True),

  ("test relate left",
   SIMPLE_SCENE,
   r"relate(unique(\x.star(x)),unique(\x.rock(x)),left)",
   False),

  ("test relate right",
   SIMPLE_SCENE,
   r"relate(unique(\x.rock(x)),unique(\x.star(x)),right)",
   False),

  ("test relate right",
   SIMPLE_SCENE,
   r"relate(unique(\x.star(x)),unique(\x.rock(x)),right)",
   True),

  ("test relate_n 1",
   SIMPLE_SCENE,
   r"relate_n(unique(\x.star(x)),unique(\x.rock(x)),right,1)",
   True),

  ("test relate_n 2",
   SIMPLE_SCENE,
   r"relate_n(unique(\x.circle(x)),unique(\x.diamond(x)),up,2)",
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
   r"in_half(unique(\x.heart(x)),up)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.heart(x)),right)",
   False),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.heart(x)),down)",
   True),

  ("test in_half",
   SIMPLE_SCENE,
   r"in_half(unique(\x.heart(x)),left)",
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
   r"is_edge(unique(\x.diamond(x)))",
   True),

  ("test is_edge",
   SIMPLE_SCENE,
   r"is_edge(unique(\x.heart(x)))",
   False),

  ("test complex query",
   SIMPLE_SCENE,
   r"horse(unique(\x.relate(x,unique(\y.diamond(y)),down)))",
   True),

  ("test complex query",
   SIMPLE_SCENE,
   r"horse(unique(\x.and_(horse(x),relate(x,unique(\y.diamond(y)),down))))",
   True),

]

EC_EXPRESSIONS = [
  "(lambda (ec_unique $0 spade))", # test basic object predicate
  "(lambda (move (ec_unique $0 spade)))", # test pick
  "(lambda (relate (ec_unique $0 spade) (ec_unique $0 puddle) down))", # test relate down
  "(lambda (relate (ec_unique $0 puddle) (ec_unique $0 spade) down))", # test relate down
  "(lambda (relate (ec_unique $0 spade) (ec_unique $0 puddle) up))", # test relate up
  "(lambda (relate (ec_unique $0 puddle) (ec_unique $0 spade) up))", # test relate up
  "(lambda (relate (ec_unique $0 rock) (ec_unique $0 star) left))", # test relate left
  "(lambda (relate (ec_unique $0 rock) (ec_unique $0 puddle) left))", # test relate left
  "(lambda (relate (ec_unique $0 rock) (ec_unique $0 star) right))", # test relate right
  "(lambda (relate (ec_unique $0 star) (ec_unique $0 rock) right))", # test relate right
  "(lambda (relate (ec_unique $0 puddle) (ec_unique $0 rock) right))", # test relate right
  "(lambda (relate_n (ec_unique $0 star) (ec_unique $0 rock) right 1))", # test relate_n 1
  "(lambda (relate_n (ec_unique $0 star) (ec_unique $0 spade) up 2))", # test relate_n 2,
  "(lambda (in_half (ec_unique $0 star) up))", # test in half
  "(lambda (in_half (ec_unique $0 star) right))", # test in half
  "(lambda (in_half (ec_unique $0 star) down))", # test in half
  "(lambda (in_half (ec_unique $0 star) left))", # test in half
  "(lambda (in_half (ec_unique $0 triangle) up))", # test in half
  "(lambda (in_half (ec_unique $0 triangle) right))", # test in half
  "(lambda (in_half (ec_unique $0 triangle) down))", # test in half
  "(lambda (in_half (ec_unique $0 triangle) left))", # test in half
  "(lambda (is_edge (ec_unique $0 circle)))", # test is_edge
  "(lambda (is_edge (ec_unique $0 star)))", # test is_edge
  "(lambda (is_edge (ec_unique $0 rock)))", # test is_edge
  "(lambda (is_edge (ec_unique $0 heart)))", # test is_edge
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

def _test_ec_case(scene, ec_expression, expected, msg=None):
  from pprint import pprint
  print("Objects:")
  pprint(scene['objects'])

  puddleworldTypes, puddleworldPrimitives = convertOntology(ec_ontology)

  expr = ec_program.Program.parse(ec_expression)
  value = expr.runWithArguments([scene])
  print(expr)
  print("Expected:", expected)
  print("Observed:", value)
  eq_(value, expected, msg)

def test_ec_fixed_cases():
  for i, (msg, scene, _, expected) in enumerate(CASES):
    scene = process_scene([scene])
    if i < len(EC_EXPRESSIONS):
      expression = EC_EXPRESSIONS[i]

      f = partial(_test_ec_case)
      f.description = "test case: %s -> %s" % (expression, expected)
      yield f, scene, expression, expected, msg





