"""
Puddleworld EC Learner.
"""
####
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../ec/")
sys.path.insert(0, "../pyccg")
sys.path.insert(0, "../pyccg/nltk")
####

import datetime
import numpy as np
import os
import random

from ec import explorationCompression, commandlineArguments, Task, ecIterator
from grammar import Grammar
from utilities import eprint, numberOfCPUs
from task import *

from puddleworldOntology import ec_ontology, process_scene
from utils import convertOntology

#### Load and prepare dataset for EC.
def loadPuddleWorldTasks(datafile='data/puddleworld.json'):
    """
    Loads a pre-processed version of the Puddleworld tasks.
    """
    import json
    with open(datafile) as f:
        result = json.load(f)
    return result

def makePuddleworldTask(raw_task, input_type, output_type):
    _, objects, instructions, goal = raw_task
    scene = process_scene(objects) # Convert into a PyCCG-style scene.
    task = Task(name=instructions,
                request=arrow(input_type, output_type),
                examples=[(scene, tuple(goal))],
                features=instructions)
    return task

 
def makeTasks(train_key, test_key, input_type, output_type):
    data = loadPuddleWorldTasks()
    raw_train, raw_test = data[train_key], data[test_key]

    # Sort by description length.
    def sort_by_instruction(dataset):
        lengths = np.array([len(instr.split(" ")) for _, _, instr, _ in dataset])
        sorted_idxs = lengths.argsort()
        return [dataset[idx] for idx in sorted_idxs]
 
    sorted_train, sorted_test = sort_by_instruction(raw_train), sort_by_instruction(raw_test)
    train, test = [makePuddleworldTask(task, input_type, output_type) for task in sorted_train], [makePuddleworldTask(task, input_type, output_type) for task in sorted_test]
    return train, test

def makeLocalTasks(input_type, output_type):
    return makeTasks('local_train', 'local_test', input_type, output_type)

def makeGlobalTasks(input_type, output_type):
    return makeTasks('global_train', 'global_test', input_type, output_type)

### Run the learner.
def puddleworld_options(parser):
    parser.add_argument(
        "--local",
        action="store_true",
        default=True,
        help='Include local navigation tasks.'
        )
    parser.add_argument(
        "--global",
        action="store_true",
        default=False,
        help='Include global navigation tasks.'
        )
    parser.add_argument("--random-seed", 
        type=int, 
        default=0
        )

if __name__ == "__main__":
    args = commandlineArguments(
        enumerationTimeout=10, 
        activation='tanh', 
        iterations=1, 
        recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        extras=puddleworld_options)

    # Set up.
    random.seed(args.pop("random_seed"))
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/puddleworld/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    # Convert ontology.
    puddleworldTypes, puddleworldPrimitives = convertOntology(ec_ontology)
    input_type, output_type = puddleworldTypes['model'], puddleworldTypes['action']

    # Make tasks.
    doLocal, doGlobal = args.pop('local'), args.pop('global')
    (localTrain, localTest) = makeLocalTasks(input_type, output_type) if doLocal else ([], [])
    (globalTrain, globalTest) = makeGlobalTasks(input_type, output_type) if doGlobal else ([], [])
    eprint("Using local tasks: %d train, %d test" % (len(localTrain), len(localTest)))
    eprint("Using global tasks: %d train, %d test" % (len(globalTrain), len(globalTest)))

