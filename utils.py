"""
utils.py
"""
#### If need to add adjacent libraries.
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../pyccg")
sys.path.insert(0, "../pyccg/nltk")
####

import ec.type as ec_type
import ec.program as ec_program
from pyccg.logic import TypeSystem, Ontology, Expression, ComplexType

def convertFunction(f, ecTypes):
	"""
	Converts a typed PyCCG function into an EC Primitive.
	:return: ec_type.Primitive
	"""
	# TODO(catwong): handle non complex types.

	returnType = ecTypes[f.return_type.name] # TODO (catwong): make sure this is correct.
	for t in f.arg_types:
		if type(t) is ComplexType:
			print("TODO (catwong) : handle complex types.")
			return None

	ecArgs = [ecTypes[t.name] for t in f.arg_types]
	ecFunction = ec_program.Primitive(f.name, arrow(*ecArgs, returnType), f.defn)

	print(ecFunction)
	return ecFunction


def convertOntology(ontology):
	"""
	Converts a PyCCG ontology to a list of Dreamcoder primitives.
    Conversion:
   		Ontology Types -> EC baseTypes
		Ontology Functions -> EC Primitives 
	:return:
		types: dict from str ontology type names -> EC baseTypes.
		primitives: list of primtitives with str names given by the ontology function names.
	"""
	# Ontology types -> EC base types.
	types = {t.name : ec_type.baseType("t_" + t.name) for t in ontology.types if t.name is not None}

	# Create primitives for any of the constants.
	
	# Ontology functions -> Typed EC primitives.
	for f in ontology.functions:
		convertFunction(f, types)
		
	# Functions have names and types
	return types, None




if __name__ == "__main__":
	print("Demo: puddleworld ontology conversion.")

	import puddleworldOntology 
	puddleworldTypes, puddleworldPrimitives = convertOntology(puddleworldOntology.ontology)

	# TODO (cathywong): print the types



