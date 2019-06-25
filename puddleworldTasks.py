"""Puddleworld tasks: Gridworld scenes of objects and NL instructions."""

import numpy as np

from task import *

from puddleworldOntology import ec_ontology, process_scene

#### Utility functions to load and prepare dataset for EC.
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
                examples=[([scene], tuple(goal))],
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
    """Make tasks with 'local' spatial relations."""
    return makeTasks('local_train', 'local_test', input_type, output_type)

def makeGlobalTasks(input_type, output_type):
    """Make tasks with 'global' spatial relations."""
    return makeTasks('global_train', 'global_test', input_type, output_type)

def makeTinyTasks(input_type, output_type, num_tiny=1, tiny_scene_size=2):
    """Make tiny scenes for bootstrapping and debugging purposes.
       Scenes are all scene_size x scene_size maps with a single object and that object name as instructions."""
    from puddleworldOntology import obj_dict

    def makeTinyTask(size):
        obj = random.Random().choice(range(len(obj_dict) - 1))
        obj_name = obj_dict[obj]
        instructions = obj_name
        objects = np.zeros((size, size))
        row, col = random.Random().choice(range(size)), random.Random().choice(range(size))
        objects[row][col] = obj
        goal = (row, col)
        task = (None,  [objects], instructions, goal)
        task = makePuddleworldTask(task, input_type, output_type)
        return task

    train, test = [makeTinyTask(tiny_scene_size) for _ in range(num_tiny)], [makeTinyTask(tiny_scene_size) for _ in range(num_tiny)]
    return train, []
