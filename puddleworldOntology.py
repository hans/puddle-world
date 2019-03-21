########
# Puddleworld Ontology: defines a type system, constants, and predicates available for use
# in logical forms.

from pyccg.logic import TypeSystem, Ontology, Expression

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
    return (target["row"], target["col"])


def fn_relate(a, b, direction):
  return fn_relate_n(a, b, direction, 1)

def fn_relate_n(a, b, direction, n):
  # a is DIRECTION of b
  if direction == "left":
    return a["row"] == b["row"] and a["col"] == b["col"] - n
  if direction == "right":
    return a["row"] == b["row"] and a["col"] == b["col"] + n
  if direction == "down":
    return a["col"] == b["col"] and a["row"] == b["row"] + n
  if direction == "up":
    return a["col"] == b["col"] and a["row"] == b["row"] - n

def fn_in_half(obj, direction):
  if direction == "left":
    return a["col"] < SCENE_WIDTH / 2
  if direction == "right":
    return a["col"] > SCENE_WIDTH / 2
  if direction == "down":
    return a["row"] > SCENE_HEIGHT / 2
  if direction == "up":
    return a["row"] < SCENE_HEIGHT / 2

def fn_max_in_dir(obj, direction):
  # e.g. "bottom most horse"
  # where "horse" forms the relevant comparison class
  lookup_keys = {
      "left": "col", "right": "col",
      "down": "row", "up": "row"
  }
  key = lookup_keys[direction]
  reverse = direction in ["left", "up"]

  comparison_class = set([obj]) # TODO critical: need global scene info here..
  return max(comparison_class, key=operator.itemgetter(key), reverse=reverse) == obj

def fn_is_edge(obj):
  # true when obj is at the edge of the grid.
  return obj["col"] in [0, SCENE_WIDTH - 1] and obj["row"] in [0, SCENE_HEIGHT - 1]


types = TypeSystem(["object", "boolean", "action", "direction", "int"])

functions = [
  types.new_function("move", ("object", "action"), fn_pick),
  types.new_function("relate", ("object", "object", "direction", "boolean"), fn_relate),
  types.new_function("relate_n", ("object", "object", "direction", "int", "boolean"), fn_relate_n),
  types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
  types.new_function("in_half", ("object", "direction", "boolean"), fn_in_half),
  types.new_function("apply", (("object", "boolean"), "object", "boolean"), lambda f, o: f(o)),
  types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
  types.new_function("max_in_dir", ("object", "direction", "boolean"), fn_max_in_dir),
  types.new_function("is_edge", ("object", "boolean"), fn_is_edge),
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
  types.new_constant("1", "int"),
]

ontology = Ontology(types, functions, constants)