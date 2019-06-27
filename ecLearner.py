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
import dill
import numpy as np
import os
import random
import string

from ec import explorationCompression, commandlineArguments, Task, ecIterator
from frontier import Frontier
from enumeration import * # EC enumeration.
from taskRankGraphs import plotEmbeddingWithLabels
from grammar import Grammar
from utilities import eprint, numberOfCPUs
from recognition import *
from task import *

from pyccg.lexicon import Lexicon
from pyccg.word_learner import WordLearner
from pyccg.logic import read_ec_sexpr

from puddleworldOntology import ec_ontology, process_scene
from puddleworldTasks import *
from utils import convertOntology, ecTaskAsPyCCGUpdate, puddleworld_ec_translation_fn


class InstructionsFeatureExtractor(RecurrentFeatureExtractor):
    """
    InstructionsFeatureExtractor: minimal EC-recogntition-model feature extractor for the instruction strings.
    """

    def _tokenize_string(self, features):
        """Ultra simple tokenizer. Removes punctuation, then splits on spaces."""
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokenized = features.translate(remove_punctuation).lower().split()
        return tokenized

    def tokenize(self, features):
        """Recurrent feature extractor expects examples in a [(xs, y)] form where xs -> a list of inputs.
           list, so match this form.
        """
        xs, y = [self._tokenize_string(features)], []
        return [(xs, y)]

    def build_lexicon(self, tasks, testingTasks):
        """Lexicon of all tokens that appear in train and test tasks."""
        lexicon = set()
        allTasks = tasks + testingTasks
        for t in allTasks:
            tokens = self._tokenize_string(t.features)
            lexicon.update(tokens)
        return list(lexicon)

    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.recomputeTasks = False # TODO(cathywong): probably want to recompute.
        self.useFeatures = True

        lexicon = self.build_lexicon(tasks, testingTasks)
        print("Lexicon len, values", len(lexicon), lexicon[:10])
        super(InstructionsFeatureExtractor, self).__init__(lexicon=lexicon,
                                                            H=64, # Hidden layer.
                                                            tasks=tasks,
                                                            bidirectional=True,
                                                            cuda=cuda)

### PyCCG Word Learner
initial_puddleworld_lex = Lexicon.fromstring(r"""
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
""", ec_ontology, include_semantics=True)

class ECLanguageLearner:
    """
    ECLanguageLearner: driver class that manages learning between PyCCG and EC.

    ec_ontology_translation_fn: runs on the EC program string if there are any function naming conversions.
    """
    def __init__(self,
                pyccg_learner,
                ec_ontology_translation_fn=None):

                self.pyccg_learner = pyccg_learner
                self.ec_ontology_translation_fn = ec_ontology_translation_fn


    def _update_pyccg_timeout(self, update, timeout):
        """
        Wraps PyCCG update with distant in a timeout.
        Returns: S-expression semantics for the sentence or None if none found within the timeout.
        """
        import signal
        def timeout_handler(signum, frame):
            raise Exception("PyCCG enumeration timeout.")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout) # Start the stopwatch.
        results = None
        try:
            instruction, model, goal = update
            results = self.pyccg_learner.update_with_distant(instruction, model, goal)
        except Exception:
            pass

        if results and len(results) > 0:
            root_token, _ = results[0][0].label()
            meaning = root_token.semantics()
        else:
            meaning = None

        return meaning


    def _update_pyccg_with_distant_batch(self, tasks, timeout):
        """
        Sequential update of PyCCG with distant batch. Returns discovered parses.
        Ret:
            pyccg_meanings: dict from task -> PyCCG S-expression semantics for the sentence, 
                            or None if no expression was found.
        """
        pyccg_meanings = {t: self._update_pyccg_timeout(ecTaskAsPyCCGUpdate(t, self.pyccg_learner.ontology), timeout) for t in tasks}
        return pyccg_meanings

    def _update_pyccg_with_supervised_batch(self, frontiers):
        """
        Sequential update of PyCCG supervised on EC frontiers.
        """
        for frontier in frontiers:
            instruction, model, goal = ecTaskAsPyCCGUpdate(frontier.task, self.pyccg_learner.ontology)
            for entry in frontier.entries:
                if self.ec_ontology_translation_fn:
                    ec_expr = str(entry.program) if self.ec_ontology_translation_fn is None else  self.ec_ontology_translation_fn(str(entry.program))
                converted = read_ec_sexpr(ec_expr)
                # TODO (catwong, jgauthier): no update with supervised.
                print("****ALERT: NOT YET IMPLEMENTED FULLY: NO UDPATE WITH SUPERVISED *****")

    def _pyccg_meanings_to_ec_frontiers(self, pyccg_meanings):
        """
        Ret:
            pyccg_frontiers: dict from task -> Dreamcoder frontiers.
        """
        return None

    def _describe_pyccg_results(self, pyccg_results):
        for task in pyccg_results:
            if pyccg_results[task]:
                print('HIT %s w/ %s' %(task.name, str(pyccg_results[task])))
            else:
                print('MISS %s' % task.name)

    def wake_generative_with_pyccg(self,
                    grammar, tasks, 
                    maximumFrontier=None,
                    enumerationTimeout=None,
                    CPUs=None,
                    solver=None,
                    evaluationTimeout=None):
        """
        Dreamcoder wake_generative using PYCCG enumeration to guide exploration.

        Enumerates from PyCCG with a timeout and blindly from the EC grammar.
        Updates PyCCG using both sets of discovered meanings.
        Converts the meanings into EC-style frontiers to be handed off to EC.
        """
        # Enumerate PyCCG meanings and update the word learner.
        pyccg_meanings = self._update_pyccg_with_distant_batch(tasks, enumerationTimeout)
       
        # Enumerate the remaining tasks using EC-style blind enumeration.
        unsolved_tasks = [task for task in tasks if pyccg_meanings[task] is None]
        fallback_frontiers, fallback_times = multicoreEnumeration(grammar, unsolved_tasks, 
                                                   maximumFrontier=maximumFrontier,
                                                   enumerationTimeout=enumerationTimeout,
                                                   CPUs=CPUs,
                                                   solver=solver,
                                                   evaluationTimeout=evaluationTimeout)
        print("PyCCG model parsing results")
        self._describe_pyccg_results(pyccg_meanings)
        print("Non-language generative model enumeration results:")
        print(Frontier.describe(fallback_frontiers))

        # Update PyCCG model with fallback discovered frontiers.
        self._update_pyccg_with_supervised_batch(fallback_frontiers) # TODO(catwong, jgauthier): does not yet update.

        # Convert and consolidate PyCCG meanings and fallback frontiers for handoff to EC.
        pyccg_frontiers = self._pyccg_meanings_to_ec_frontiers(pyccg_meanings)
        all_frontiers = {t : pyccg_frontiers[t] if t in pyccg_frontiers else fallback_frontiers[t]}
        all_times = {t : enumerationTimeout if t in pyccg_frontiers else fallback_times[t]}

        return all_frontiers, all_times
            

### Additional command line arguments for Puddleworld.
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
    parser.add_argument(
        "--tiny",
        action="store_true",
        default=False,
        help='Include tiny tasks.'
        )
    parser.add_argument(
        "--num_tiny",
        default=1,
        type=int,
        help='How many tiny tasks to create.'
        )
    parser.add_argument(
        "--tiny_scene_size",
        default=1,
        type=int,
        help='Size of tiny scenes; will be NxN scenes.'
        )
    parser.add_argument("--random-seed", 
        type=int, 
        default=0
        )
    parser.add_argument("--checkpoint-analysis",
        default=None,
        type=str)

if __name__ == "__main__":
    # EC command line arguments.
    args = commandlineArguments(
        enumerationTimeout=10, 
        activation='tanh', 
        iterations=1, 
        recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        featureExtractor=InstructionsFeatureExtractor,
        extras=puddleworld_options)

    checkpoint_analysis = args.pop("checkpoint_analysis") # EC checkpoints need to be run out of their calling files, so this is here.

    """Run the EC learner."""
    if checkpoint_analysis is None:
        # Set up output directories.
        random.seed(args.pop("random_seed"))
        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "experimentOutputs/puddleworld/%s"%timestamp
        os.system("mkdir -p %s"%outputDirectory)

        # Convert pyccg ontology -> Dreamcoder.
        puddleworldTypes, puddleworldPrimitives = convertOntology(ec_ontology)
        input_type, output_type = puddleworldTypes['model'], puddleworldTypes['action']

        # Convert sentences-scenes -> Dreamcoder style tasks.
        doLocal, doGlobal, doTiny= args.pop('local'), args.pop('global'), args.pop('tiny')
        num_tiny, tiny_size = args.pop('num_tiny'), args.pop('tiny_scene_size')

        (localTrain, localTest) = makeLocalTasks(input_type, output_type) if doLocal else ([], [])
        (globalTrain, globalTest) = makeGlobalTasks(input_type, output_type) if doGlobal else ([], [])
        (tinyTrain, tinyTest) = makeTinyTasks(input_type, output_type, num_tiny, tiny_size) if doTiny else ([], [])
        allTrain, allTest = localTrain + globalTrain + tinyTrain, localTest + globalTest + tinyTest
        eprint("Using local tasks: %d train, %d test" % (len(localTrain), len(localTest)))
        eprint("Using global tasks: %d train, %d test" % (len(globalTrain), len(globalTest)))
        eprint("Using tiny tasks of size %d: %d train, %d test" % (tiny_size, len(tinyTrain), len(tinyTest)))
        eprint("Using total tasks: %d train, %d test" % (len(allTrain), len(allTest)))

        # Make Dreamcoder grammar.
        baseGrammar = Grammar.uniform(puddleworldPrimitives)
        print(baseGrammar.json())



        # Initialize the language learner driver.
        ##### DEBUGGING (cathy)
        # (note: cathy - to debug, jankily run with made up arguments, but we don't actually wanna do that,)
        pyccg_learner = WordLearner(initial_puddleworld_lex)
        learner = ECLanguageLearner(pyccg_learner, ec_ontology_translation_fn=puddleworld_ec_translation_fn)

        tasks, maximumFrontier, enumerationTimeout, CPUs, solver, evaluationTimeout = localTrain[:10], 2, 2, 5, 'python', 5
        learner.wake_generative_with_pyccg(baseGrammar, tasks, 
                    maximumFrontier,
                    enumerationTimeout,
                    CPUs,
                    solver,
                    evaluationTimeout)

        assert False
        ##### DEBUGGING (cathy)
        ##### DEBUGGING UNCOMMENT BELOW WHEN DONE (cathy)


        # Run Dreamcoder exploration/compression.
        # explorationCompression(baseGrammar, allTrain, 
        #                         testingTasks=allTest, 
        #                         outputPrefix=outputDirectory, **args)

    





    ### Checkpoint analyses. Can be safely ignored to run the learner itself.
    # These are in this file because Dill is silly and requires loading from the original calling file.
    if checkpoint_analysis is not None:
        # Load the checkpoint.
        print("Loading checkpoint ", checkpoint_analysis)
        with open(checkpoint_analysis,'rb') as handle:
            result = dill.load(handle)
            recognitionModel = result.recognitionModel

        def plotTSNE(title, labels_embeddings):
            """Plots TSNE. labels_embeddings = dict from string labels -> embeddings"""
            from sklearn.manifold import TSNE
            tsne = TSNE(random_state=0, perplexity=5, learning_rate=50, n_iter=10000)
            labels = list(labels_embeddings.keys())
            embeddings = list(labels_embeddings[label] for label in labels_embeddings)
            print("Clustering %d embeddings of shape: %s" % (len(embeddings), str(embeddings[0].shape)))
            labels, embeddings = np.array(labels), np.array(embeddings)
            clustered = tsne.fit_transform(embeddings)
            plotEmbeddingWithLabels(clustered, 
                                        labels, 
                                        title, 
                                        os.path.join("%s_tsne_labels.png" % title.replace(" ", ""))) # TODO(catwong): change output to take commandline. 

        # Get the recurrent feature extractor symbol embeddings.
        plotSymbolEmbeddings = False
        if plotSymbolEmbeddings:
            symbolEmbeddings = result.recognitionModel.featureExtractor.symbolEmbeddings()
            plotTSNE("Symbol embeddings", symbolEmbeddings)
        
        # Plot the word-specific log productions.
        plotWordHiddenState = True
        if plotWordHiddenState:
            # Get the lexicon and turn them into 'tasks.'
            lexicon = [word for word in result.recognitionModel.featureExtractor.lexicon if word not in result.recognitionModel.featureExtractor.specialSymbols]
            print(lexicon)
            lexicon_tasks = [Task(name=None,
                request=None,
                examples=[([1],[1])],
                features=word) for word in lexicon]

            # Hacky! Turn words into tasks to reuse existing code that extracts out the productions from tasks.
            symbolEmbeddings = result.recognitionModel.taskGrammarFeatureLogProductions(lexicon_tasks)
            plotTSNE("word_log_productions", symbolEmbeddings)

        # Get other layers in the recognition model task (sentence-level) embeddings.
        plotTaskEmbeddings = False
        if plotTaskEmbeddings:
            def get_task_embeddings(result, embedding_key):
                task_embeddings = {}
                for task in result.recognitionTaskMetrics:
                    if embedding_key in result.recognitionTaskMetrics[task].keys():
                        task_name = task.name
                        task_embedding = result.recognitionTaskMetrics[task][embedding_key]
                        task_embeddings[task_name] = task_embedding
                if len(task_embeddings.keys()) == 0:
                    print("No embeddings for key found, ", embedding_key)
                    assert False
                return task_embeddings
            
            embedding_key = 'taskLogProductions'
            task_embeddings = get_task_embeddings(result, embedding_key)
            plotTSNE(embedding_key, task_embeddings)

            embedding_key = 'hiddenState'
            task_embeddings = get_task_embeddings(result, embedding_key)
            plotTSNE(embedding_key, task_embeddings)

            # If heldout task log productions
            embedding_key = 'heldoutTaskLogProductions'
            task_embeddings = get_task_embeddings(result, embedding_key)
            plotTSNE(embedding_key, task_embeddings)




