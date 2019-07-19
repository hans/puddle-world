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
from pyccg.word_learner import *

from dreamcoder.ec import Task
from dreamcoder.type import arrow
import dreamcoder.program as ec_program

from puddleworldOntology import make_puddleworld_ontology, process_scene, \
    puddleworld_ec_pyccg_translation_fn, puddleworld_pyccg_ec_translation_fn
from utils import ecTaskAsPyCCGUpdate, convertOntology

ontology = make_puddleworld_ontology('pyccg')
pyccg_model_ontology = make_puddleworld_ontology('pyccg_model')
ec_ontology = make_puddleworld_ontology('default')

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

  ("test basic object predicate",
   SIMPLE_SCENE,
   r"unique(diamond)",
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
  "(lambda (ec_unique_p $0 diamond_p))", # test basic object predicate
  "(lambda (ec_unique_p $0 diamond_p))", # test basic object predicate
  "(lambda (move_p (ec_unique_p $0 diamond_p)))", # test pick
  "(lambda (relate_p (ec_unique_p $0 diamond_p) (ec_unique_p $0 star_p) down_p))", # test relate down
  "(lambda (relate_p (ec_unique_p $0 star_p) (ec_unique_p $0 diamond_p) down_p))", # test relate down
  "(lambda (relate_p (ec_unique_p $0 diamond_p) (ec_unique_p $0 star_p) up_p))", # test relate up
  "(lambda (relate_p (ec_unique_p $0 star_p) (ec_unique_p $0 diamond_p) up_p))", # test relate up
  "(lambda (relate_p (ec_unique_p $0 rock_p) (ec_unique_p $0 star_p) left_p))", # test relate left
  "(lambda (relate_p (ec_unique_p $0 star_p) (ec_unique_p $0 rock_p) left_p))", # test relate left
  "(lambda (relate_p (ec_unique_p $0 rock_p) (ec_unique_p $0 star_p) right_p))", # test relate right
  "(lambda (relate_p (ec_unique_p $0 star_p) (ec_unique_p $0 rock_p) right_p))", # test relate right
  "(lambda (relate_n_p (ec_unique_p $0 star_p) (ec_unique_p $0 rock_p) right_p 1_p))", # test relate_n 1
  "(lambda (relate_n_p (ec_unique_p $0 circle_p) (ec_unique_p $0 diamond_p) up_p 2_p))", # test relate_n 2,
  "(lambda (in_half_p (ec_unique_p $0 star_p) up_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 star_p) right_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 star_p) down_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 star_p) left_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 heart_p) up_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 heart_p) right_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 heart_p) down_p))", # test in half
  "(lambda (in_half_p (ec_unique_p $0 heart_p) left_p))", # test in half
  "(lambda (is_edge_p (ec_unique_p $0 circle_p)))", # test is_edge
  "(lambda (is_edge_p (ec_unique_p $0 star_p)))", # test is_edge
  "(lambda (is_edge_p (ec_unique_p $0 diamond_p)))", # test is_edge
  "(lambda (is_edge_p (ec_unique_p $0 heart_p)))", # test is_edge
]

def test_model_based_scene():
  TINY_SCENE = np.array(
    [[0.0, 0.0,
    0.0, 6.0]])
  expr = Expression.fromstring(r"\x. ec_unique(x, \y.diamond(y))")
  scene = process_scene([TINY_SCENE])
  model_object = frozendict(objects=tuple(scene['objects']), type='model')
  wrapped_scene = {'objects' : [model_object]}
  expected = frozendict(row=2, col=9, type="diamond")

  model = Model(wrapped_scene, pyccg_model_ontology)
  value = model.evaluate(expr)
  print(expr)
  print("Expected:", expected)
  print("Observed:", value)

  eq_(value, expected, msg=None)


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

def test_ec_task_as_pyccg_scene():
  instruction = 'go to the diamond'
  scene = process_scene([SIMPLE_SCENE])
  goal = (2,9)

  puddleworldTypes, puddleworldPrimitives = convertOntology(ec_ontology)
  ec_task = Task(name=instruction,
                request=arrow(puddleworldTypes['model'], puddleworldTypes['action']),
                examples=[([scene], tuple(goal))],
                features=instruction)

  converted_instruction, converted_model, converted_goal = ecTaskAsPyCCGUpdate(ec_task, ontology)

  eq_(converted_instruction, instruction.split())
  eq_(converted_model.scene, scene)
  eq_(converted_goal, goal)

def test_puddleworld_pyccg_ec_translation():
  """
  pyccg->EC ontology conversion should correctly manage special case predicate
  `unique`
  """
  cases = [
    (r"unique(\z1.horse(z1))", "(lambda (ec_unique_p $0 (lambda (horse_p $0))))"),
    (r"relate_n(unique(\x.star(x)),unique(\x.rock(x)),right,1)", "(lambda (relate_n_p (ec_unique_p $0 (lambda (star_p $0))) (ec_unique_p $0 (lambda (rock_p $0))) right_p 1_p))")
  ]

  def _do_case(pyccg_lf_str, ec_lf_str):
    pyccg_lf = Expression.fromstring(pyccg_lf_str)
    ontology.typecheck(pyccg_lf)
    converted_pyccg = puddleworld_pyccg_ec_translation_fn(pyccg_lf, ontology)
    print("Converted %s->%s" % (pyccg_lf_str, converted_pyccg))
    eq_(converted_pyccg, ec_lf_str)

  for pyccg_str, ec_str in cases:
    yield _do_case, pyccg_str, ec_str


def test_puddleworld_ec_pyccg_translation():
  """
  EC->pyccg ontology should conversion should correctly manage special case
  predicate `unique`
  """
  cases = [
    ("(lambda (ec_unique_p $0 (lambda (horse_p $0))))", r"unique(\z1.horse(z1))"),
    ("(lambda (relate_n_p (ec_unique_p $0 (lambda (star_p $0))) (ec_unique_p $0 (lambda (rock_p $0))) right_p 1_p))", r"relate_n(unique(\z1.star(z1)),unique(\z2.rock(z2)),right,1)")
  ]
  def _do_case(ec_lf_str, pyccg_lf_str):
    expr = puddleworld_ec_pyccg_translation_fn(ec_lf_str, ontology)
    eq_(str(expr), pyccg_lf_str)

  for ec_str, pyccg_str in cases:
    yield _do_case, ec_str, pyccg_str


def test_iter_expressions():
  """
  Make sure some complicated expressions are available in expr iteration
  """
  exprs = ontology.iter_expressions(type_request=ontology.types["object", "object", "action"], max_depth=6)
  exprs = set(str(x) for x in exprs)
  ok_(r"\z2 z1.move(unique(\z3.relate(z3,z2,z1)))" in exprs)

def test_update_with_supervised():
  from puddleworldOntology import SEED_PUDDLEWORLD_LEX

  answer = Expression.fromstring(r'move(unique(\z1.diamond(z1)))')
  sentence = "go to diamond".split()
  scene = process_scene([SIMPLE_SCENE])

  learner = WordLearner(SEED_PUDDLEWORLD_LEX)
  model = Model(scene, ontology)
  learner.update_with_supervision(sentence, learner, answer)


