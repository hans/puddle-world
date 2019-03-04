#! /usr/bin/env python3
import pickle

from frozendict import frozendict
import numpy as np
from tqdm import tqdm

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.model import Model
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression
from pyccg.word_learner import WordLearner


#########
# Load and prepare dataset.

obj_dict = {
  0: 'grass',
  1: 'puddle',
  2: 'star',
  3: 'circle',
  4: 'triangle',
  5: 'heart',
  6: 'spade',
  7: 'diamond',
  8: 'rock',
  9: 'tree',
  10: 'house',
  11: 'horse'
}

def process_scene(states, objects):
  objects = objects[0]
  objects = {(x, y): frozendict(x=x, y=y, type=obj_dict[val])
              for (y, x), val in np.ndenumerate(objects)}
  return {"objects": list(objects.values())}


with open("local_human_train.pkl", "rb") as data_f:
  dataset = pickle.load(data_f)
states, objects, instructions, goals = dataset[0], dataset[1], dataset[4], dataset[6]

# order by increasing length
lengths = np.array([len(instr.split(" ")) for instr in instructions])
sorted_idxs = lengths.argsort()
states = states[sorted_idxs]
objects = objects[sorted_idxs]
instructions = [instructions[idx] for idx in sorted_idxs]
SCENE_WIDTH, SCENE_HEIGHT = objects[0][0].shape


########
# Ontology: defines a type system, constants, and predicates available for use
# in logical forms.


class Object(object):
  def __init__(self, shape, size, material):
    self.shape = shape
    self.size = size
    self.material = material

  def __hash__(self):
    return hash((self.shape, self.size, self.material))

  def __eq__(self, other):
    return other.__class__ == self.__class__ and hash(self) == hash(other)

  def __str__(self):
    return "Object(%s, %s, %s)" % (self.shape, self.size, self.material)


def fn_unique(xs):
  # print(xs)
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]


def fn_exists(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  return len(true_xs) > 0


class Action(object):
  pass

class Move(Action):
  def __init__(self, target):
    self.target = target

  def __hash__(self):
    return hash(tuple("move", self.target))
  def __str__(self):
    return "Move(%s)" % self.target


def fn_pick(target):
  if isinstance(target, frozendict):
    return (target["x"], target["y"])


def fn_relate(a, b, direction):
  if direction == "left":
    return a["x"] < b["x"]
  if direction == "right":
    return a["x"] > b["y"]
  if direction == "down":
    return a["x"] == b["x"] and a["y"] == b["y"] - 1
    # return a["y"] > b["y"]
  if direction == "up":
    return a["y"] < b["y"]

def fn_in_half(obj, direction):
  if direction == "left":
    return a["x"] < SCENE_WIDTH / 2
  if direction == "right":
    return a["x"] > SCENE_WIDTH / 2
  if direction == "down":
    return a["y"] > SCENE_WIDTH / 2
  if direction == "up":
    return a["y"] < SCENE_WIDTH / 2


types = TypeSystem(["object", "boolean", "action", "direction"])

functions = [
  types.new_function("move", ("object", "action"), fn_pick),
  types.new_function("relate", ("object", "object", "direction", "boolean"), fn_relate),
  types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
  types.new_function("in_half", ("object", "direction", "boolean"), fn_in_half),
  types.new_function("apply", (("object", "boolean"), "object", "boolean"), lambda f, o: f(o)),
  types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
]
def make_obj_fn(obj):
  return lambda o: o["type"] == obj
functions.extend([types.new_function(obj, ("object", "boolean"), make_obj_fn(obj))
                  for obj in obj_dict.values()])

constants = [
  types.new_constant("true", "boolean"),
  types.new_constant("left", "direction"),
  types.new_constant("right", "direction"),
  types.new_constant("up", "direction"),
  types.new_constant("down", "direction"),
]

ontology = Ontology(types, functions, constants)


#######
# Lexicon: defines an initial set of word -> (syntax, meaning) mappings.
# Weights are initialized uniformly by default.


initial_lex = Lexicon.fromstring(r"""
  :- S, N

  reach => S/N {\x.move(x)}
  below => S/N {\x.move(unique(\y.relate(x,y,down)))}
  above => S/N {\x.move(unique(\y.relate(x,y,up)))}

  , => S\S/S {\a b.a}
  , => S\S/S {\a b.b}

  of => N\N/N {\x d y.relate(x,y,d)}
  of => N\N/N {\x d y.relate(unique(x),d,y)}
  to => N\N/N {\x y.x}

  one => S/N/N {\x d.move(unique(\y.relate(x,y,d)))}
  right => N/N {\f x.and_(apply(f, x),in_half(x,right))}

  the => N/N {\x.unique(x)}

  left => N {left}
  below => N {down}
  right => N {right}
  horse => N {\x.horse(x)}
  rock => N {\x.rock(x)}
  cell => N {\x.true}
  spade => N {\x.spade(x)}
  spade => N {unique(\x.spade(x))}
  heart => N {\x.heart(x)}
  circle => N {\x.circle(x)}
""", ontology, include_semantics=True)
initial_lex.debug_print()

p = WeightedCCGChartParser(initial_lex)
printCCGDerivation(p.parse("one below the rock".split())[0])
printCCGDerivation(p.parse("below spade".split())[0])



learner = WordLearner(initial_lex)
instructions = instructions[:3]
for states_i, objects_i, instruction_i, goal_i in tqdm(zip(states, objects, instructions, goals), total=len(instructions)):
  print(instruction_i)
  scene = process_scene(states_i, objects_i)
  model = Model(scene, ontology)
  print(goal_i)

  # TESTS = [
  #     r"unique(\x.spade(x))",
  #     r"move(unique(\x.spade(x)))",
  #     r"unique(\y.relate(unique(\x.spade(x)),y,down))",
  #     r"move(unique(\y.relate(unique(\x.spade(x)),y,down)))",
  # ]
  # for test in TESTS:
  #   print(test, model.evaluate(Expression.fromstring(test)))
  # sys.exit()

  instruction_i = instruction_i.split()
  results = learner.update_with_distant(instruction_i, model, goal_i)

  if results:
    printCCGDerivation(results[0][0])

  break


print("Lexicon:")
learner.lexicon.debug_print()
print()

