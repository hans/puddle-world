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
from pyccg.logic import TypeSystem, Ontology, Expression, ComplexType, BasicType


def convertType(t, ecTypes):
	"""Converts PyCCG basic and complex types."""
	if type(t) is BasicType:
		return ecTypes[t.name]
	elif type(t) is ComplexType:
		return ec_type.arrow(convertType(t.first, ecTypes), convertType(t.second, ecTypes))


def convertFunction(f, ecTypes):
	"""Converts a typed PyCCG function into an EC Primitive."""
	ecArgs = [convertType(t, ecTypes) for t in f.arg_types]
	ecReturn = convertType(f.return_type, ecTypes)
	ecFunction = ec_program.Primitive(f.name, ec_type.arrow(*ecArgs, ecReturn), f.defn)

	return ecFunction


def convertOntology(ontology):
	"""
	Converts a PyCCG ontology to a list of Dreamcoder primitives.
    Conversion:
   		Ontology Types -> EC baseTypes
   		Ontology Constants -> EC Primitive (null-valued.)
		Ontology Functions -> EC Primitives 
	:return:
		types: dict from str ontology type names -> EC baseTypes.
		primitives: list of primtitives with str names given by the ontology function names.
	"""
	# Ontology types -> EC base types.
	types = {t.name : ec_type.baseType("t_" + t.name) for t in ontology.types if t.name is not None}

	# TODO (cathywong): is this how we want to handle constants.
	constants = [ec_program.Primitive(c.name, convertType(c.type, types), None) for c in ontology.constants]
	
	# Ontology functions -> Typed EC primitives.
	functions = [convertFunction(f, types) for f in ontology.functions]		
		
	# Functions have names and types
	return types, constants + functions




if __name__ == "__main__":
	print("Demo: puddleworld ontology conversion.")

	import puddleworldOntology 
	puddleworldTypes, puddleworldPrimitives = convertOntology(puddleworldOntology.ontology)

	print("Converted %d types: " % len(puddleworldTypes))
	for t in puddleworldTypes:
		print("New base type: %s -> %s" % (str(t), str(puddleworldTypes[t])))

	print("Converted %d Primitives: " % len(puddleworldPrimitives))
	for p in puddleworldPrimitives:
		if p.value is None:
			print("New Primitive from Constant: %s : %s" % (str(p), str(p.tp)))
		else:
			print("New Primitive from Function: %s : %s" % (str(p), str(p.tp)))



