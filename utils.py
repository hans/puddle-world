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
	Conversion:
		Ontology Types -> EC baseTypes
		Ontology Constants -> EC Primitive (null-valued.)
		Ontology Functions -> EC Primitives 
	:return:
		types: dict from ontology type names -> EC baseTypes.
		primitives: list of primtitives with names given by the ontology functions and constants.
	"""
	types = {t.name : ec_type.baseType("t_" + t.name) for t in ontology.types if t.name is not None}

	# TODO (cathywong): is this how we want to handle constants.
	constants = [ec_program.Primitive(c.name, convertType(c.type, types), None) for c in ontology.constants]
	
	functions = [convertFunction(f, types) for f in ontology.functions]	
		
	return types, constants + functions


if __name__ == "__main__":
	print("Demo: puddleworld ontology conversion.")

	from puddleworldOntology import ontology
	puddleworldTypes, puddleworldPrimitives = convertOntology(ontology)

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
		obj1 = {'type' : 'circle'}
		obj2 = {'type' : 'spade'}

		## Single arity functions.
		# obj_fn
		p = ec_program.Program.parse("(lambda (circle $0))")
		print(p)
		print("Object: %s , Eval: %r" % (obj1['type'], p.runWithArguments([obj1])))
		print("Object: %s , Eval: %r" % (obj2['type'], p.runWithArguments([obj2])))
		print("\n")

		# move
		target = {'row': 0, 'col': 0}
		p = ec_program.Program.parse("(lambda (move $0))")
		print(p)
		print("Eval: %s" % str(p.runWithArguments([target])))
		print("\n")

		## Double arity functions.
		p = ec_program.Program.parse("(lambda (lambda (and_ $0 $1)))")
		print(p)
		print(p.runWithArguments([True, False]))

		p = ec_program.Program.parse("(and_ (lambda (spade $0)) (lambda (spade $1)))")
		print(p)
		print(p.runWithArguments([obj1, obj2]))



