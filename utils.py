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

import type as ec_type
import program as ec_program
from utilities import curry

from pyccg.chart import WeightedCCGChartParser, printCCGDerivation
from pyccg.lexicon import Lexicon
from pyccg.logic import TypeSystem, Ontology, Expression, ComplexType, BasicType

#### EC <-> PYCCG Ontology Conversion
def convertType(t, ecTypes):
	"""Converts PyCCG basic and complex types -> EC Types."""
	if type(t) is BasicType:
		return ecTypes[t.name]
	elif type(t) is ComplexType:
		return ec_type.arrow(convertType(t.first, ecTypes), convertType(t.second, ecTypes))


def convertFunction(f, ecTypes):
	"""Converts a typed PyCCG function -> EC Primitive."""
	ecArgs = [convertType(t, ecTypes) for t in f.arg_types]
	ecReturn = convertType(f.return_type, ecTypes)

	ecFunction = ec_program.Primitive(f.name, ec_type.arrow(*ecArgs, ecReturn), curry(f.defn))

	return ecFunction


def convertOntology(ontology):
	"""
	Converts a PyCCG ontology to a list of Dreamcoder primitives.
	Adds a tmodel and fn_model_evaluate function, in keeping with PyCCG's model evaluation framework.

	Conversion:
		Ontology Types -> EC baseTypes
		Ontology Constants -> EC Primitive (null-valued.)
		Ontology Functions -> EC Primitives 
	:return:
		types: dict from ontology type names -> EC baseTypes.
		primitives: list of primtitives with names given by the ontology functions and constants.
	"""
	types = {t.name : ec_type.baseType("t_" + t.name) for t in ontology.types if t.name is not None}

	constants = [ec_program.Primitive(c.name, convertType(c.type, types), c.name) for c in ontology.constants]
	
	# Check for EC versions of these functions.
	functions = []
	function_names = { f.name : f for f in ontology.functions }
	for f in ontology.functions:
		if 'ec_'+ f.name in function_names.keys():
			pass
		else:
			functions.extend([convertFunction(f, types)])
	
	return types, constants + functions

if __name__ == "__main__":
	print("Demo: puddleworld ontology conversion.")
	import numpy as np
	from puddleworldOntology import ec_ontology, process_scene
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

	if True:
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

		





