"""
utils.py
"""
#### If need to add adjacent libraries.
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../ec/")
sys.path.insert(0, "../pyccg")
sys.path.insert(0, "../pyccg/nltk")
####
import string

import type as ec_type
import program as ec_program
from utilities import curry

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.model import Model
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression, ComplexType, BasicType

#### PyCCG Ontology <-> EC Ontology Conversion
def convertType(t, ecTypes):
    """Converts PyCCG basic and complex types -> EC Types."""
    if type(t) is BasicType:
        return ecTypes[t.name]
    elif type(t) is ComplexType:
        return ec_type.arrow(convertType(t.first, ecTypes), convertType(t.second, ecTypes))

def convertFunction(f, ecTypes, namespace):
    """Converts a typed PyCCG function -> EC Primitive."""
    ecArgs = [convertType(t, ecTypes) for t in f.arg_types]
    ecReturn = convertType(f.return_type, ecTypes)

    ecFunction = ec_program.Primitive(f.name + namespace, ec_type.arrow(*ecArgs, ecReturn), curry(f.defn))

    return ecFunction


def convertOntology(ontology, namespace='_p'):
    """
    Converts a PyCCG ontology to a list of Dreamcoder primitives.
    Adds a tmodel and fn_model_evaluate function, in keeping with PyCCG's model evaluation framework.

    For namespace reasons: appends a namespace to the end of all converted things.
    Conversion:
        Ontology Types -> EC baseTypes
        Ontology Constants -> EC Primitive (null-valued.)
        Ontology Functions -> EC Primitives 
    :return:
        types: dict from ontology type names -> EC baseTypes.
        primitives: list of primtitives with names given by the ontology functions and constants.
    """
    types = {t.name : ec_type.baseType("t_" + t.name + namespace) for t in ontology.types if t.name is not None}

    constants = [ec_program.Primitive(c.name + namespace, convertType(c.type, types), c.name) for c in ontology.constants]
    
    # Check for EC versions of these functions.
    functions = []
    function_names = { f.name : f for f in ontology.functions }
    for f in ontology.functions:
        if 'ec_'+ f.name in function_names.keys():
            pass
        else:
            functions.extend([convertFunction(f, types, namespace)])
    
    return types, constants + functions

def getOCamlDefinitions(ont_types, ont_primitives, ontology_name=""):
    """
    Utility function to output all the necessary type and primitive definitions for a given type and primitive set.
    These will need to be put in solvers/program.ml to use the Dreamcoder OCaml compiler.

    ont_types, ont_primitives: the output of convertOntology(ontology).
    """
    print("(* %s Type Definitions *)" % ontology_name)
    for t in ont_types:
        print('let %s = make_ground "%s";;' % (ont_types[t], ont_types[t]))

    print("(* %s Primitive Definitions *)" % ontology_name)
    for p in ont_primitives:
        # Creates primitive function names with thunk type definitions.
        prim_name, prim_signature = str(p), str(p.tp)
        ocaml_signature = prim_signature.replace('->', '@>')
        print('ignore(primitive "%s" (%s) (fun x -> x));;' % (prim_name, ocaml_signature))


### EC to PyCCG convenience conversions
def ecTaskAsPyCCGUpdate(task, ontology):
    """
    Converts an EC task into a PyCCG update.
    Assumes the task has a single scene and the instructions as a string.
    :return:
        tokenized_instruction, model, goal.
    """
    remove_punctuation = str.maketrans('', '', string.punctuation)
    tokenized = task.features.translate(remove_punctuation).lower().split()
    
    scene, goal = task.examples[0] # Unpack the examples.
    scene, goal = scene[0], goal
    return tokenized, Model(scene, ontology), goal

### Other convenience functions.
def filter_tasks_mlu(tasks, mlu):
    """
    Returns only tasks up to a set maximum instruction length.
    Uses a 'tokenizer' just by splitting on spaces.
    """
    
    # For convenience, also alphabetize within each description length.
    mlu_tasks = []
    for utterance_len in range(mlu+1):
        utterances = [t for t in tasks if len(t.features.split(" ")) == utterance_len]
        utterances = sorted(utterances, key = lambda t: t.features)
        mlu_tasks += utterances
    return mlu_tasks

    #return [t for t in tasks if len(t.features.split(" ")) <= mlu]

if __name__ == "__main__":
    print("Demo: puddleworld ontology conversion.")
    import numpy as np
    from puddleworldOntology import ec_ontology, process_scene, obj_dict
    puddleworldTypes, puddleworldPrimitives = convertOntology(ec_ontology)

    if True:
        print("Converted %d types: " % len(puddleworldTypes))
        for t in puddleworldTypes:
            print("New base type: %s -> %s" % (str(t), str(puddleworldTypes[t])))

        print("Converted %d Primitives: " % len(puddleworldPrimitives))
        for p in puddleworldPrimitives:
            if p.value is None:
                print("New Primitive from Constant: %s : %s" % (str(p), str(p.tp)))
            else:
                print("New Primitive from Function: %s : %s" % (str(p), str(p.tp)))

        print("\nOCaml Type Definitions")
        getOCamlDefinitions(puddleworldTypes, puddleworldPrimitives, ontology_name="Puddleworld")

    if False:
        import random
        size = 2
        print("Relate n on tiny tasks.")
        obj = random.Random().choice(range(len(obj_dict) - 1))
        obj_name = obj_dict[obj]
        instructions = obj_name 
        objects = np.zeros((size, size))
        row, col = random.Random().choice(range(size)), random.Random().choice(range(size))
        objects[row][col] = obj
        goal = (row, col)
        scene = process_scene([objects]) # Convert into a PyCCG-style scene.
        print(scene)

        fns = [
            '(lambda (move (ec_unique $0 (lambda (relate_n $0 (ec_unique $1 (lambda (puddle $0))) down 1)))))',
        ]
        for lambd in fns:
            p = ec_program.Program.parse(lambd)
            print(p)
            print("Eval: %s" % str(p.runWithArguments([scene])))
            print("\n")

    if False:
        print("Demo: EC2-style evaluations debug DSL.")
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
        
        scene = process_scene([SIMPLE_SCENE])

        p = ec_program.Program.parse('(lambda (move_debug $0))')
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        p = ec_program.Program.parse('(lambda (move_debug2 $0))')
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        p = ec_program.Program.parse('(lambda (move (ec_unique $0 is_obj)))')
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        p = ec_program.Program.parse('(lambda (move (ec_unique $0 (lambda (is_obj $0)))))')
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        p = ec_program.Program.parse('(lambda (move (ec_unique $0 (lambda (is_obj $0)))))')
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print(p.evaluate([])('test'))
        print("\n")

    if False:
        print("Demo: EC2-style evaluations after conversion.")
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
        scene = process_scene([SIMPLE_SCENE])

        ## Single arity functions.
        # obj_fn
        obj1 = scene['objects'][0]
        obj2 = scene['objects'][9]
        p = ec_program.Program.parse("(lambda (grass $0))")
        print(p)
        print("Object: %s , Eval: %r" % (obj1['type'], p.runWithArguments([obj1])))
        print("Object: %s , Eval: %r" % (obj2['type'], p.runWithArguments([obj2])))
        print("\n")

        # Test move.
        obj0 = scene['objects'][0]
        p = ec_program.Program.parse("(lambda (move $0))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([obj0])))
        print("\n")

        ## Double arity functions.
        p = ec_program.Program.parse("(lambda (lambda (and_ $0 $1)))")
        print(p)
        print(p.runWithArguments([True, False]))
        print("\n")

        p = ec_program.Program.parse("(lambda (lambda (and_ $1 (star $0))))")
        print(p)
        print(p.runWithArguments([True, obj2]))
        print("\n")
        
        # Test basic object predicate.
        p = ec_program.Program.parse("(lambda (ec_unique $0 spade))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test pick.
        p = ec_program.Program.parse("(lambda (move (ec_unique $0 spade)))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test relate down.
        p = ec_program.Program.parse("(lambda (relate (ec_unique $0 spade) (ec_unique $0 puddle) down))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test relate down.
        p = ec_program.Program.parse("(lambda (relate (ec_unique $0 puddle) (ec_unique $0 spade) down))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test relate up.
        p = ec_program.Program.parse("(lambda (relate (ec_unique $0 spade) (ec_unique $0 puddle) up))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test relate up.
        p = ec_program.Program.parse("(lambda (relate (ec_unique $0 puddle) (ec_unique $0 spade) up))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")

        # Test relate n
        p = ec_program.Program.parse("(lambda (relate_n (ec_unique $0 star) (ec_unique $0 rock) right 1))")
        print(p)
        print("Eval: %s" % str(p.runWithArguments([scene])))
        print("\n")
        




