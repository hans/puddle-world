#! /usr/bin/env python3
import logging
logging.basicConfig(level=logging.DEBUG)
import operator
import pickle
import sys

from frozendict import frozendict
import numpy as np
from tqdm import tqdm

#### Add EC dependency
sys.path.append("../ec")
from puddleworldTasks import loadPuddleWorldTasks

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.model import Model
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression
from pyccg.word_learner import WordLearner

from puddleworldOntology import ontology, process_scene


#########
# Load and prepare dataset.

dataset = loadPuddleWorldTasks("data/puddleworld.json")
dataset = dataset["local_train"]

lengths = np.array([len(instr.split(" ")) for _, _, instr, _ in dataset])
sorted_idxs = lengths.argsort()


#######
# Lexicon: defines an initial set of word -> (syntax, meaning) mappings.
# Weights are initialized uniformly by default.


initial_lex = Lexicon.fromstring(r"""
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
  # triangle => N {\x.triangle(x)}
""", ontology, include_semantics=True)
initial_lex.debug_print()

p = WeightedCCGChartParser(initial_lex)



learner = WordLearner(initial_lex)
i = 100
from pprint import pprint
for idx in sorted_idxs:
  _, objects_i, instruction_i, goal_i = dataset[idx]
  goal_i = tuple(goal_i)

  if instruction_i != "one above triangle":
    continue

  print(i)
  print(instruction_i)

  scene = process_scene(objects_i)
  model = Model(scene, ontology)

  print(np.array(objects_i))
  print(goal_i)

  instruction_i = instruction_i.split()
  results = learner.update_with_distant(instruction_i, model, goal_i)

  if results:
    printCCGDerivation(results[0][0])

  assert False

print("Lexicon:")
learner.lexicon.debug_print()
print()

