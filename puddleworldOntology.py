########
# Puddleworld Ontology: defines a type system, constants, and predicates available for use
# in logical forms.

from frozendict import frozendict
import numpy as np

from pyccg import logic as l
from pyccg.lexicon import Lexicon

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
    except Exception as e:
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


## Build ontologies.
def make_puddleworld_ontology(ontology_type='pyccg'):
  """
  ontology_type:
    pyccg: includes all of the functions above, for pyccg.
    pyccg_model: includes an alternate unique function that only runs on models.
    default: extends the ontology with a Model-typed EC unique.
    relate_n: adds more n and removes 'relate' to encourage use of 'relate_n'
  """
  # Add types.
  type_names = ["object", "boolean", "action", "direction", "int"]
  type_names.extend(['model']) # For EC enumeration on grounded scenes
  types = l.TypeSystem(type_names)

  def make_obj_fn(obj):
    return lambda o: o["type"] == obj

  functions = [
    types.new_function("move", ("object", "action"), fn_pick),
    types.new_function("relate_n", ("object", "object", "direction", "int", "boolean"), fn_relate_n),
    types.new_function("unique", (("object", "boolean"), "object"), fn_unique),
    types.new_function("in_half", ("object", "direction", "boolean"), fn_in_half),
    types.new_function("apply", (("object", "boolean"), "object", "boolean"), lambda f, o: f(o)),
    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda a, b: a and b),
    types.new_function("max_in_dir", ("object", "direction", "boolean"), fn_max_in_dir),
    types.new_function("is_edge", ("object", "boolean"), fn_is_edge),
  ]
  functions.extend([types.new_function(obj, ("object", "boolean"), make_obj_fn(obj))
                    for obj in obj_dict.values()])

  if ontology_type != 'relate_n':
    functions.extend([types.new_function("relate", ("object", "object", "direction", "boolean"), fn_relate)])

  constants = [
    types.new_constant("true", "boolean"),
    types.new_constant("left", "direction"),
    types.new_constant("right", "direction"),
    types.new_constant("up", "direction"),
    types.new_constant("down", "direction"),
    types.new_constant("1", "int"),
    types.new_constant("2", "int"),
  ]

  if ontology_type == 'pyccg':
    pass
  elif ontology_type == 'default' or ontology_type == 'relate_n' or ontology_type == 'pyccg_model':
    functions.extend([types.new_function("ec_unique", ("model", ("object", "boolean"), "object"), ec_fn_unique)])
  elif ontology_type == 'relate_n':
    constants.extend([
      types.new_constant("3", "int"),
      types.new_constant("4", "int"),
      types.new_constant("5", "int"),
    ])
  else:
    raise Exception("Invalid ontology type %s" % ontology_type)
  return l.Ontology(types, functions, constants)

### Make initial lexicon
SEED_PUDDLEWORLD_LEX = Lexicon.fromstring(r"""
  :- S:N
  reach => S/N {\x.move(x)}
  reach => S/N {\x.move(unique(x))}
  below => S/N {\x.move(unique(\y.relate(y,x,down)))}
  above => S/N {\x.move(unique(\y.relate(y,x,up)))}

  , => S\S/S {\a b.a}
  , => S\S/S {\a b.b}
  
  of => N\N/N {\x d y.relate(x,y,d)}
  of => N\N/N {\x d y.relate(unique(x),d,y)}
  to => N\N/N {\x y.x}

  one => S/N/N {\d x.move(unique(\y.relate(y,x,d)))}
  one => S/N/N {\d x.move(unique(\y.relate_n(y,x,d,1)))}
  right => N/N {\f x.and_(apply(f, x),in_half(x,right))}

  most => N\N/N {\x d.max_in_dir(x, d)}

  the => N/N {\x.unique(x)}

  left => N {left}
  below => N {down}
  above => N {up}
  right => N {right}
  horse => N {\x.horse(x)}
  rock => N {\x.rock(x)}
  rock => N {unique(\x.rock(x))}
  cell => N {\x.true}
  spade => N {\x.spade(x)}
  spade => N {unique(\x.spade(x))}
  heart => N {\x.heart(x)}
  heart => N {unique(\x.heart(x))}
  circle => N {\x.circle(x)}
  triangle => N {\x.triangle(x)}
""", make_puddleworld_ontology(ontology_type='pyccg'), include_semantics=True)

def process_scene(scene_objects):
  """
  Convert puddle-world object array into a representation compatible with this
  ontology.
  """
  scene_objects = scene_objects[0]
  scene_objects = {(row, col): frozendict(row=row, col=col, type=obj_dict[val])
                   for (row, col), val in np.ndenumerate(scene_objects)}
  return {"objects": list(scene_objects.values())}


STATEFUL_PREDICATES = ["unique"]

def puddleworld_ec_pyccg_translation_fn(raw_expr, ontology, namespace='_p', ec_fn_tag='ec_'):
    """
    Convenience translation function to remove Puddleworld namespacing before conversion
    to S-expr, or add it back.
    """
    raw_expr = raw_expr.replace(namespace+" ", " ")
    raw_expr = raw_expr.replace(namespace+")", ")")
    raw_expr = raw_expr.replace(ec_fn_tag, "")
    expr, _ = ontology.read_ec_sexpr(raw_expr, typecheck=False)

    # Remove context variable and its spot in stateful predicates.
    assert isinstance(expr, l.LambdaExpression)
    world_variable, expr = expr.variable, expr.term
    def visit(node):
        if isinstance(node, l.ApplicationExpression):
            if node.pred.variable.name in STATEFUL_PREDICATES:
                node.remove_argument(0)
            for i, arg in enumerate(node.args):
                visit(arg)
        elif isinstance(node, l.LambdaExpression):
            visit(node.term)
    visit(expr)

    expr = expr.normalize()
    ontology.typecheck(expr)
    return expr


def puddleworld_pyccg_ec_translation_fn(lf, ontology, namespace="_p", ec_fn_tag="ec_"):
    """
    Convert puddleworld pyccg sentence-level meanings to EC sentence-level meanings.
    """
    # First handle stateful predicates. These are explicitly represented as
    # extra first arguments in EC expressions.
    world_variable = l.Variable("w")
    # insert into any stateful predicates as first argument

    namespaced_strings = [str(function) for function in ontology.functions_dict]
    namespaced_strings += [str(constant) for constant in ontology.constants_dict]

    def rename_ec_namespace(name):
      namespaced_strings = [str(function) for function in ontology.functions_dict]
      namespaced_strings += [str(constant) for constant in ontology.constants_dict]
      orig_name = name 

      if orig_name in namespaced_strings:
        name += namespace
      if orig_name in STATEFUL_PREDICATES:
        name = ec_fn_tag + name
      return name

    def visit(node):
        if isinstance(node, l.ConstantExpression):
          node.variable.name = rename_ec_namespace(node.variable.name)
        if isinstance(node, l.ApplicationExpression):
            # Insert first argument.
            if node.pred.variable.name in STATEFUL_PREDICATES:
                node.insert_argument(0, l.IndividualVariableExpression(world_variable))
            # Namespacing.
            node.pred.variable.name = rename_ec_namespace(node.pred.variable.name)
            for i, arg in enumerate(node.args):
                visit(arg)
        elif isinstance(node, l.LambdaExpression):
            visit(node.term)

    visit(lf)

    lf = l.LambdaExpression(world_variable, lf)

    ret_str = ontology.as_ec_sexpr(lf)

    return ret_str
    

####
# A tiny DSL for debugging purposes.
def fn_pick_debug(model):
  return (0, 0)

def fn_pick_debug2(model):
  return (0, 0)

# ec_functions_debug = [
#     types.new_function("ec_unique", ("model", ("object", "boolean"), "object"), ec_fn_unique),
#     types.new_function("move_debug", ("model", "action"), fn_pick_debug),
#     types.new_function("move_debug2", ("model", "action"), fn_pick_debug2),
#     types.new_function("is_obj", ("object", "boolean"), lambda o : True),
#     types.new_function("move", ("object", "action"), fn_pick),
# ]
