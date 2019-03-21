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

	import puddleworldOntology 
	puddleworldTypes, puddleworldPrimitives = convertOntology(puddleworldOntology.ontology)

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
		print("Demo: EC2 evaluation.")
		obj1 = {'type' : 'circle'}
		obj2 = {'type' : 'spade'}

		# obj_fn
		p = ec_program.Program.parse("(lambda (circle $0))")
		print(p)
		print("Object: %s , Eval: %r" % (obj1['type'], p.runWithArguments([obj1])))
		print("Object: %s , Eval: %r" % (obj2['type'], p.runWithArguments([obj2])))

		# AND
		p = ec_program.Program.parse("(lambda (lambda (and_ circle $0 circle $1)))")
		print(p.runWithArguments([obj1, obj2]))

		# Unique
		p = ec_program.Program.parse("(lambda (unique circle $0))")
		print(p)

		


	if False:
		# TODO(catwong): hold off on this until refactored by Jon.
		print("Demo: PyCCG Lexicon parsing.")
		initial_lex = Lexicon.fromstring(r"""
		  :- S, N

		  reach => S/N {\x.move(x)}
		  reach => S/N {\x.move(unique(x))}
		  below => S/N {\x.move(unique(\y.relate(x,y,down)))}
		  above => S/N {\x.move(unique(\y.relate(x,y,up)))}

		  , => S\S/S {\a b.a}
		  , => S\S/S {\a b.b}

		  of => N\N/N {\x d y.relate(x,y,d)}
		  of => N\N/N {\x d y.relate(unique(x),d,y)}
		  to => N\N/N {\x y.x}

		  one => S/N/N {\x d.move(unique(\y.relate(x,y,d)))}
		  one => S/N/N {\x d.move(unique(\y.relate_n(x,y,d,1)))}
		  right => N/N {\f x.and_(apply(f, x),in_half(x,right))}

		  most => N\N/N {\x d.max_in_dir(x, d)}

		  the => N/N {\x.unique(x)}

		  left => N {left}
		  below => N {down}
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
		""", puddleworldOntology.ontology, include_semantics=True)
		initial_lex.debug_print()

		p = WeightedCCGChartParser(initial_lex)
		# printCCGDerivation(p.parse("the rock".split())[0])
		# printCCGDerivation(p.parse("one below the rock".split())[0])
		# printCCGDerivation(p.parse("below spade".split())[0])

		# Test evaluation on a given scene.

