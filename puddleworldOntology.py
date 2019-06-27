########
# Puddleworld Ontology: defines a type system, constants, and predicates available for use
# in logical forms.

from frozendict import frozendict
import numpy as np

from pyccg.logic import TypeSystem, Ontology, Expression

SCENE_WIDTH = 10
SCENE_HEIGHT = 10

obj_dict = {
  0: 'grass',
  -1: 'puddle',
  1: 'star',
  2: 'circle',
  3: 'triangle',
  4: 'heart',
  5: 'spade',
  6: 'diamond',
  7: 'rock',
  8: 'tree',
  9: 'house',
  10: 'horse'
}

def ec_fn_tmodel_evaluate(model, expr):
  """Generic evaluation function to evaluate expression on a PyCCG-style domain."""
  cf = {}
  for u in model['objects']:
    try:
      val = expr(u)
    except:
      val = False
    cf[u] = val
  return cf

def ec_fn_unique(model, expr):
  cf = ec_fn_tmodel_evaluate(model, expr)
  return fn_unique(cf)

def ec_fn_exists(model, expr):
  cf = ec_fn_tmodel_evaluate(model, expr)
  return fn_exists(cf)

def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

def fn_exists(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  return len(true_xs) > 0


def fn_pick(target):
  if isinstance(target, frozendict): 
    return (target["row"], target["col"])


def fn_relate(a, b, direction):
  return fn_relate_n(a, b, direction, 1)

def fn_relate_n(a, b, direction, n):
  # a is DIRECTION of b
  n = int(n)
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
    return obj["col"] < SCENE_WIDTH / 2
  if direction == "right":
    return obj["col"] > SCENE_WIDTH / 2
  if direction == "down":
    return obj["row"] > SCENE_HEIGHT / 2
  if direction == "up":
    return obj["row"] < SCENE_HEIGHT / 2

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
  return obj["col"] in [0, SCENE_WIDTH - 1] or obj["row"] in [0, SCENE_HEIGHT - 1]


type_names = ["object", "boolean", "action", "direction", "int"]
type_names.extend(['model']) # For EC enumeration on grounded scenes
types = TypeSystem(type_names)
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
  types.new_constant("2", "int"),
]

ontology = Ontology(types, functions, constants)

# Add model-typed versions of function for EC.
ec_functions = functions
ec_functions.extend([
  types.new_function("ec_unique", ("model", ("object", "boolean"), "object"), ec_fn_unique)
  ])
ec_ontology = Ontology(types, ec_functions, constants)


def process_scene(scene_objects):
  """
  Convert puddle-world object array into a representation compatible with this
  ontology.
  """
  scene_objects = scene_objects[0]
  scene_objects = {(row, col): frozendict(row=row, col=col, type=obj_dict[val])
                   for (row, col), val in np.ndenumerate(scene_objects)}
  return {"objects": list(scene_objects.values())}


def puddleworld_ec_translation_fn(raw_expr, is_pyccg_to_ec, namespace='_p'):
    """
    Convenience translation function to remove Puddleworld namespacing before conversion
    to S-expr, or add it back.
    """
    if is_pyccg_to_ec:
        # Namespace all of the functions and constants
        namespaced_strings = [function for function in ontology.functions_dict]
        namespaced_strings += [str(constant) for constant in ontology.constants_dict]
        for renameable in namespaced_strings:
          raw_expr = raw_expr.replace(renameable, renameable+namespace)
        return raw_expr
    else:
        return raw_expr.replace(namespace+" ", " ")


####
# A tiny DSL for debugging purposes.
def fn_pick_debug(model):
  return (0, 0)

def fn_pick_debug2(model):
  return (0, 0)

ec_functions_debug = [
    types.new_function("ec_unique", ("model", ("object", "boolean"), "object"), ec_fn_unique),
    types.new_function("move_debug", ("model", "action"), fn_pick_debug),
    types.new_function("move_debug2", ("model", "action"), fn_pick_debug2),
    types.new_function("is_obj", ("object", "boolean"), lambda o : True),
    types.new_function("move", ("object", "action"), fn_pick),
]
