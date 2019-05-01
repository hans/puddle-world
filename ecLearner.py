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
import os
import random

from ec import explorationCompression, commandlineArguments, Task, ecIterator
from grammar import Grammar
from utilities import eprint, numberOfCPUs

#### Load and prepare dataset for EC.
def makePuddleworldTasks()

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

	# Make tasks.
	doLocal, doGlobal = args.pop('local'), args.pop('global')
	localTrain, localTest = makeLocalTasks() if doLocal else [], []
	globalTrain, globalTest = makeGlobalTasks() if doGlobal else [], []
	eprint("Using local tasks: %d train, %d test" % (len(localTrain), len(localTest)))
	eprint("Using global tasks: %d train, %d test" % (len(globalTrain), len(globalTest)))

