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
from collections import Counter, defaultdict

from bin.taskRankGraphs import plotEmbeddingWithLabels

from dreamcoder.ec import explorationCompression, commandlineArguments, Task, ecIterator
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.enumeration import * # EC enumeration.
from dreamcoder.grammar import ContextualGrammar, Grammar
from dreamcoder.program import Program
from dreamcoder.utilities import eprint, numberOfCPUs
from dreamcoder.recognition import *
from dreamcoder.task import *

from pyccg.lexicon import Lexicon
from pyccg.word_learner import WordLearner

from puddleworldOntology import make_puddleworld_ontology, process_scene, puddleworld_ec_pyccg_translation_fn, puddleworld_pyccg_ec_translation_fn, SEED_PUDDLEWORLD_LEX
from puddleworldTasks import *
from utils import convertOntology, ecTaskAsPyCCGUpdate, filter_tasks_mlu, MLUTaskBatcher


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

        print("Built lexicon from %d tasks, found %d words" % (len(allTasks), len(lexicon)))
        print(sorted(list(lexicon)))
        return list(lexicon)

    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.recomputeTasks = False # TODO(cathywong): probably want to recompute.
        self.useFeatures = True

        lexicon = self.build_lexicon(tasks, testingTasks)
        super(InstructionsFeatureExtractor, self).__init__(lexicon=lexicon,
                                                            H=32, # Hidden layer.
                                                            tasks=tasks,
                                                            bidirectional=True,
                                                            cuda=cuda)

class ECLanguageLearner:
    """
    ECLanguageLearner: driver class that manages learning between PyCCG and EC.

    pyccg2ec_translation: convert PyCCG expressions to EC if there are any ontology/namespacing differences.
    ec2pyccg_translation: convert EC expressions to PyCCG if there are any ontology/namespacing differences.
    use_pyccg_enum: if True: use PyCCG parsing to discover sentence frontiers.
    use_blind_enum: if True: to use blind enumeration on unsolved frontiers.
    word_reweighting: options to reweight the grammar based on the tasks seen so far.
    starting_grammar: the original grammar, which can be used for grammar reweighting to emphasize original primitives.
    """
    def __init__(self,
                pyccg_learner,
                pyccg2ec_translation=None,
                ec2pyccg_translation=None,
                use_pyccg_enum=False,
                use_blind_enum=False,
                word_reweighting=None,
                starting_grammar=None):

                self.pyccg_learner = pyccg_learner
                self.pyccg2ec_translation = pyccg2ec_translation
                self.ec2pyccg_translation = ec2pyccg_translation
                self.use_pyccg_enum = use_pyccg_enum
                self.use_blind_enum = use_blind_enum

                self.word_reweighting = word_reweighting
                self.starting_grammar = set(starting_grammar.primitives) # Names of all starting primitives.
                self.encountered_tasks = set() # All tasks encountered thus far.
                self.token_pseudocount = 1.0
                self.smoothing_constant = 1.0 # How much to smooth based on new words.

    def _tokenize_string(self, features):
        """Ultra simple tokenizer. Removes punctuation, then splits on spaces."""
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokenized = features.translate(remove_punctuation).lower().split()
        return tokenized

    def _reweight_unsolved_words(self, tasks, result):
        """
        Calculates a task reweighting factor proportional to potentially 'informative' words, which can be used
        to increase enumeration time, # of frontiers stored, etc.

        Reweighting factor = log((word count in unsolved tasks) / (word count in solved tasks + pseudocount)),
                            where a current batch of tasks is always considered unsolved.

        :args: 
            tasks: batch of tasks.
            result: a previous ECResult, which contains information on unsolved tasks.
        :ret: dict of {task : reweighting scores.}
        """
        # Calculate word reweighting scores for all existing tokens.
        unsolved_tokens, solved_tokens, all_tokens = defaultdict(float), defaultdict(float), set()
        for f in result.allFrontiers.values():
            if f.empty and f.task.name in self.encountered_tasks:
                for token in self._tokenize_string(f.task.features):
                     unsolved_tokens[token] += 1
                     all_tokens.add(token)
            if not f.empty:
                for token in self._tokenize_string(f.task.features):
                     solved_tokens[token] += 1
                     all_tokens.add(token)

        if len(solved_tokens) == 0: return {task : 0.0 for task in tasks} # If no solved tasks yet, continue.

        word_reweight_scores = {token: np.log(unsolved_tokens[token] + self.token_pseudocount)
            / (solved_tokens[token] + self.token_pseudocount) for token in all_tokens}
        print("Using unsolved word-based reweighting, reweighting scores.")
        print(sorted(word_reweight_scores.items(), key=lambda k: k[1], reverse=True))

        # Reweight for individual tasks.
        task_reweightings = {}
        for t in tasks:
            reweighting_factor = sum([word_reweight_scores[token] for token in self._tokenize_string(t.features)])
            print(t.name, reweighting_factor)
            # Hacky: adds the normalizing constant directly to the log likelihoods as a pseudocount.
            task_reweightings[t] = reweighting_factor

        return task_reweightings

    def _word_reweightings(self, tasks, result):
        """
        Returns a dict of {task : reweighting scores} based on task features.
        """
        if self.word_reweighting is None:
            return {task : 0.0 for task in tasks}
        elif self.word_reweighting == 'unsolved_words':
            return self._reweight_unsolved_words(tasks, result)
        else:
            raise Exception("Unknown word reweighting scheme: %s " % self.word_reweighting)

    def _update_pyccg_timeout(self, update, timeout):
        """
        Wraps PyCCG update with distant in a timeout.
        Returns: list of (S-expression semantics, logProb) tuples found for the sentence within the timeout.
        """
        import multiprocessing
        
        instruction, model, goal = update
        def update_in_timeout(instruction, model, goal, return_dict):
            return_dict['results'] = []
            return_dict['results'] = self.pyccg_learner.update_with_distant(instruction, model, goal)

        
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=update_in_timeout, args=(instruction, model, goal, return_dict))
        p.start()
        p.join(timeout)
        if p.is_alive():
            print("Timeout for t=%d on: %s" % (timeout, instruction))
            p.terminate()
            p.join()

        weighted_meanings = []
        results = return_dict['results']
        if results and len(results) > 0:
            for result in results:
                log_probability = result[1]
                root_token, _ = result[0].label()
                meaning = root_token.semantics()
                weighted_meanings.append((meaning, log_probability))
        return weighted_meanings


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
                if self.ec2pyccg_translation:
                    converted_pyccg = self.ec2pyccg_translation(str(entry.program), self.pyccg_learner.ontology)
                else:
                    converted_pyccg = self.pyccg_learner.ontology.read_ec_sexpr(ec_expr)
                print("Updating PyCCG with supervised on %s " % str(converted_pyccg))
                self.pyccg_learner.update_with_supervision(instruction, model, converted_pyccg)

    def _pyccg_meanings_to_ec_frontiers(self, pyccg_meanings):
        """
        Ret:
            pyccg_frontiers: dict from task -> Dreamcoder frontiers for tasks solved by PyCCG.
        """
        pyccg_frontiers = {}
        for task in pyccg_meanings:
            if len(pyccg_meanings[task]) > 0:
                frontier_entries = []
                for (meaning, log_prob) in pyccg_meanings[task]:
                    if self.pyccg2ec_translation:
                        ec_sexpr = self.pyccg2ec_translation(meaning, self.pyccg_learner.ontology)
                    else:
                        ec_sexpr = self.pyccg_learner.ontology.as_ec_sexpr(meaning)

                    # Uses the p=1.0 likelihood for programs that solve the task.
                    frontier_entry = FrontierEntry(
                        program=Program.parse(ec_sexpr),
                        logPrior=log_prob, 
                        logLikelihood=0.0)
                    frontier_entries.append(frontier_entry)

                pyccg_frontiers[task] = Frontier(frontier_entries, task)
        return pyccg_frontiers

    def _describe_pyccg_results(self, pyccg_results):
        for task in pyccg_results:
            if len(pyccg_results[task]) > 0:
                best_program, best_prob = pyccg_results[task][0]
                print('HIT %s w/ %s, logProb = %s' %(task.name, str(best_program), str(best_prob)))
            else:
                print('MISS %s' % task.name)

    def wake_generative_with_pyccg(self,
                    grammar, tasks, 
                    maximumFrontier=None,
                    enumerationTimeout=None,
                    CPUs=None,
                    solver=None,
                    evaluationTimeout=None,
                    result=None):
        """
        Dreamcoder wake_generative using PYCCG enumeration to guide exploration.

        Enumerates from PyCCG with a timeout and blindly from the EC grammar.
        Updates PyCCG using both sets of discovered meanings.
        Converts the meanings into EC-style frontiers to be handed off to EC.
        """
        # Store the encountered task names.
        self.encountered_tasks.update([t.name for t in tasks])

        # Enumerate PyCCG meanings and update the word learner.
        pyccg_meanings = {t : [] for t in tasks}
        if self.use_pyccg_enum:
            pyccg_meanings = self._update_pyccg_with_distant_batch(tasks, enumerationTimeout)
       
        # Enumerate the remaining tasks using EC-style blind enumeration.
        unsolved_tasks = [task for task in tasks if len(pyccg_meanings[task]) == 0]
        fallback_frontiers, fallback_times = [], None
        if self.use_blind_enum:
            task_reweightings = self._word_reweightings(unsolved_tasks, result)
            if self.word_reweighting:
                fallback_frontiers, fallback_times = multicoreEnumeration(grammar, unsolved_tasks, 
                                                           maximumFrontier=maximumFrontier,
                                                           enumerationTimeout=enumerationTimeout,
                                                           CPUs=CPUs,
                                                           solver=solver,
                                                           evaluationTimeout=evaluationTimeout,
                                                           task_reweighting=task_reweightings)
            else:
                # Backward compatibility with unmodified multicoreEnumeration.
                fallback_frontiers, fallback_times = multicoreEnumeration(grammar, unsolved_tasks, 
                                                           maximumFrontier=maximumFrontier,
                                                           enumerationTimeout=enumerationTimeout,
                                                           CPUs=CPUs,
                                                           solver=solver,
                                                           evaluationTimeout=evaluationTimeout)
        
        # Convert to EC frontiers.
        pyccg_frontiers = self._pyccg_meanings_to_ec_frontiers(pyccg_meanings)

        # Log enumeration results.
        print("PyCCG model parsing results")
        if self.use_pyccg_enum: print(Frontier.describe(pyccg_frontiers.values()))
        print("Non-language generative model enumeration results:")
        print(Frontier.describe(fallback_frontiers))

        # Update PyCCG model with fallback discovered frontiers.
        if self.use_pyccg_enum: self._update_pyccg_with_supervised_batch(fallback_frontiers) 

        assert False

        # Convert and consolidate PyCCG meanings and fallback frontiers for handoff to EC.
        
        fallback_frontiers = {frontier.task : frontier for frontier in fallback_frontiers}
        all_frontiers = {t : pyccg_frontiers[t] if t in pyccg_frontiers else fallback_frontiers[t] for t in tasks}
        all_times = {t : enumerationTimeout if t in pyccg_frontiers else fallback_times[t] for t in tasks}

        return list(all_frontiers.values()), all_times
            

### Additional command line arguments for Puddleworld.
def puddleworld_options(parser):
    # PyCCG + Dreamcoder arguments.
    parser.add_argument(
        "--disable_pyccg_enum",
        dest="use_pyccg_enum",
        action="store_false",
        help='Whether to disable PyCCG to enumerate sentence parses.'
        )
    parser.add_argument(
        "--disable_blind_enum",
        dest="use_blind_enum",
        action="store_false",
        help='Whether to disable blind multicore enumeration to enumerate sentence parses.'
        )
    parser.add_argument(
        "--mlu_cap",
        default=None,
        type=int,
        help='If provided, cap tasks by MLU.'
        )
    parser.add_argument(
        "--max_expr_depth",
        default=None,
        type=int,
        help='Max expression depth for PyCCG.'
        )
    parser.add_argument(
        "--mlu_compress",
        default=None,
        action="store_true",
        help='If provided, compress tasks by MLU.'
        )
    parser.add_argument(
        "--word_reweighting",
        default=None,
        type=str,
        help='Reweights enumeration time and frontier lengths based on task features. Options: [None, unsolved_words].'
        )

    # Puddleworld-specific.
    parser.add_argument(
        "--ontology",
        default='default',
        help='Which ontology: [default, relate_n]',
        type=str,
        )
    parser.add_argument(
        "--use_initial_lexicon",
        action="store_true",
        help='Initialize PyCCG learner with a predefined initial lexicon.'
        )
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

    # Set up output directories.
    random.seed(args.pop("random_seed"))
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/puddleworld/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    # Convert pyccg ontology -> Dreamcoder.

    ontology_type = args.pop('ontology')
    puddleworldOntology = make_puddleworld_ontology(ontology_type)

    puddleworldTypes, puddleworldPrimitives = convertOntology(puddleworldOntology)
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

    # Cap tasks by maximum length utterance (using a tokenizer)
    mlu_cap = args.pop('mlu_cap')
    if mlu_cap:
        eprint("Filtering by Maximum MLU: %d" % mlu_cap)
        allTrain, allTest = filter_tasks_mlu(allTrain, mlu_cap), filter_tasks_mlu(allTest, mlu_cap)

    eprint("Using total tasks: %d train, %d test" % (len(allTrain), len(allTest)))

    """Run the EC learner."""
    if checkpoint_analysis is None:
        # Make Dreamcoder grammar.
        baseGrammar = Grammar.uniform(puddleworldPrimitives)
        print(baseGrammar.json())

        # Initialize the language learner driver.
        use_pyccg_enum, use_blind_enum = args.pop('use_pyccg_enum'), args.pop('use_blind_enum')
        print("Using PyCCG enumeration: %s, using blind enumeration: %s" % (str(use_pyccg_enum), str(use_blind_enum)))
        
        PYCCG_MAX_EXPR_DEPTH = 4
        import logging
        logger = logging.getLogger()
        logger.disabled = True

        max_expr_depth = args.pop('max_expr_depth')
        max_expr_depth = max_expr_depth if max_expr_depth is not None else PYCCG_MAX_EXPR_DEPTH
        if args.pop('use_initial_lexicon'):
            print("Using initial lexicon for Puddleworld PyCCG learner.")
            pyccg_learner = WordLearner(SEED_PUDDLEWORLD_LEX, max_expr_depth=max_expr_depth)
        else:
            pyccg_learner = WordLearner(None, max_expr_depth=max_expr_depth)

        word_reweighting = args.pop('word_reweighting')
        learner = ECLanguageLearner(pyccg_learner, 
            pyccg2ec_translation=puddleworld_pyccg_ec_translation_fn,
            ec2pyccg_translation=puddleworld_ec_pyccg_translation_fn,
            use_pyccg_enum=use_pyccg_enum,
            use_blind_enum=use_blind_enum,
            word_reweighting=word_reweighting,
            starting_grammar=baseGrammar)

        # Initialize any task batchers for the curriculum.
        mlu_compress = args.pop('mlu_compress')
        if mlu_compress:
            task_batcher = MLUTaskBatcher()
            batcher_fn = task_batcher.getTaskBatch
        else:
            batcher_fn = None

        # Run Dreamcoder exploration/compression.
        explorationCompression(baseGrammar, allTrain, 
                                testingTasks=allTest, 
                                outputPrefix=outputDirectory, 
                                custom_wake_generative=learner.wake_generative_with_pyccg,
                                custom_task_batcher=batcher_fn,
                                **args)

    


    ###################################################################################################  
    ### Checkpoint analyses. Can be safely ignored to run the PyCCG+Dreamcoder learner itself.
    # These are in this file because Dill is silly and requires loading from the original calling file.
    def plotTSNE(title, labels_embeddings, key):
            """Plots TSNE. labels_embeddings = dict from string labels -> embeddings"""
            from sklearn.manifold import TSNE
            tsne = TSNE(random_state=0, perplexity=5, learning_rate=50, n_iter=10000)
            labels = list(labels_embeddings.keys())
            embeddings = list(labels_embeddings[label][key] for label in labels)
            print("Clustering %d embeddings of shape: %s" % (len(embeddings), str(embeddings[0].shape)))
            labels, embeddings = np.array(labels), np.array(embeddings)
            clustered = tsne.fit_transform(embeddings)
            file_name = os.path.join(outputDirectory, "%s_tsne_labels.png" % title.replace(" ", ""))
            print("Saving file to: %s" % file_name)
            plotEmbeddingWithLabels(clustered, 
                                        labels, 
                                        title, 
                                        file_name)

    def kNN(probe_dict, labels_embeddings, title, key, k=5, closest_n=None):
        """Calculates kNN for each item in labels_embeddings. labels_embeddings = dict from string labels -> embeddings.
        Returns the probe_dict with the updated nearest neighbors, distances, and the solution program associated
        with that embedding if it exists.

        closest_n: if None, returns all kNN. If int, adds only the top N (based on closeness to nearest neighbor.)
        """
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute')
        labels = list(labels_embeddings.keys())
        embeddings = [labels_embeddings[label][key] for label in labels]
        print("kNN over %d embeddings of shape: %s" % (len(embeddings), str(embeddings[0].shape)))
        labels, embeddings = np.array(labels), np.array(embeddings)
        neighbors.fit(embeddings)

        # Calculate the KNN embeddings, distances, and frontiers if applicable.
        probe_labels = list(probe_dict.keys())
        knn_results = {}
        for probe_label in probe_labels:
            if key in probe_dict[probe_label]:
                knn_results[probe_label] = {}
                neighbor_dists, neighbor_inds = neighbors.kneighbors(probe_dict[probe_label][key].reshape(1, -1), return_distance=True)
                neighbor_labels, neighbor_dists = list(labels[neighbor_inds].squeeze()), list(neighbor_dists.squeeze())
                neighbor_frontiers = [] # Task solutions associated with the queried embeddings.
                for neighbor_label in neighbor_labels:
                    if 'frontiers' in labels_embeddings[neighbor_label] and labels_embeddings[neighbor_label]['frontiers'] > 0:
                        neighbor_frontiers.append(labels_embeddings[neighbor_label]['best_program'])
                    else:
                        neighbor_frontiers.append("")
                knn_results[probe_label] = (neighbor_labels , neighbor_dists, neighbor_frontiers)

        closest_n = closest_n if closest_n is not None else len(knn_results)
        closest_n_probes = sorted(list(knn_results.keys()), key=lambda probe: knn_results[probe][1][0])[:closest_n] # Sort by nearest neighbor distance
        
        # Update the probe dict.
        for label in closest_n_probes:
            probe_dict[label]['kNN_%s_%s' % (key, title)] = knn_results[label]
        return probe_dict


    ### Run the checkpoint analysis.
    if checkpoint_analysis is not None:
        # Load the checkpoint.
        print("Loading checkpoint ", checkpoint_analysis)
        with open(checkpoint_analysis,'rb') as handle:
            result = dill.load(handle)
            recognitionModel = result.recognitionModel

        # Loads training and testing tasks, and marks them if they were solved.
        trainingUtterances = {t.features : {'frontiers': 0, 'best_program' : ("", -1.0)} for t in allTrain}
        for t in result.allFrontiers:
            trainingUtterances[t.features]['frontiers'] += len(result.allFrontiers[t].entries)
            # Add the best program for that task.
            if not result.allFrontiers[t].empty:
                bestEntry = result.allFrontiers[t].topK(1).entries[0]
                if -bestEntry.logPosterior > trainingUtterances[t.features]['best_program'][1]:
                    trainingUtterances[t.features]['best_program'] = (str(bestEntry.program), -bestEntry.logPosterior)

        unsolvedUtterances = {t : trainingUtterances[t] for t in trainingUtterances if trainingUtterances[t]['frontiers'] < 1}
        trainingUtterances = {t : trainingUtterances[t] for t in trainingUtterances if trainingUtterances[t]['frontiers'] > 0}

        # Manual analogies to probe concept learning.
        obj_strs = ['star', 'triangle', 'diamond', 'spade', 'heart', 'circle', 'heart', 'horse', 'house', 'rock', 'tree']
        ANALOGIES = []
        for obj1 in obj_strs:
            for obj2 in obj_strs:
                if obj1 != obj2:
                    for direction in ['left of', 'right of', 'above', 'below']:
                        ANALOGIES += [('reach the %s' % obj1, 'reach %s the %s' % (direction, obj1), 'reach the %s' % obj2)]
                        ANALOGIES += [('reach %s the %s' % (direction, obj1), 'reach two %s the %s' % (direction, obj1), 'reach %s the %s' % (direction, obj2))] 
                        for direction2 in ['left of', 'right of', 'above', 'below']:
                            if direction != direction2:
                                ANALOGIES += [('reach %s the %s' % (direction, obj1), 'reach %s the %s' % (direction2, obj1), 'reach %s the %s' % (direction, obj2))]
        manualAnalogyUtterances = {}
        for a, b, c in ANALOGIES:
            manualAnalogyUtterances[a], manualAnalogyUtterances[b], manualAnalogyUtterances[c] = {}, {}, {} 
        for direction in ['left', 'right', 'above', 'below']:
            manualAnalogyUtterances['reach %s' % direction] = {}

        lexiconEmbeddings = {token : {'counts' : 0} for token in result.recognitionModel.featureExtractor.lexicon}
        directLexiconEmbeddings = {token : {'counts' : 0} for token in result.recognitionModel.featureExtractor.lexicon}

        # Run the utterances through the checkpoint recognition model.
        for i, utteranceDict in enumerate([trainingUtterances, unsolvedUtterances, manualAnalogyUtterances, directLexiconEmbeddings]):
            for u in utteranceDict:
                try:
                    #### Utterance embeddings: full-sentence embeddings.
                    features_of_task = result.recognitionModel.featureExtractor.forward(u)
                    # Feature extractor utterance embeddings. Outputs directly from the feature extractor that are passed to 
                    # the recognition model.
                    utteranceDict[u]['embed_feature_extractor'] = features_of_task.data.cpu().numpy()
                    # Contextual grammar log productions.
                    features = result.recognitionModel._MLP(features_of_task)
                    utteranceDict[u]['embed_contextual_transition_matrix'] = result.recognitionModel.grammarBuilder.transitionMatrix(features).view(-1).data.cpu().numpy()
                
                    #### Calculate the token-specific embeddings.
                    for token in result.recognitionModel.featureExtractor._tokenize_string(u):
                        lexiconEmbeddings[token]['counts'] += 1
                        for embedding_key in ('embed_feature_extractor', 'embed_contextual_transition_matrix'):
                            if embedding_key not in lexiconEmbeddings[token]:
                                lexiconEmbeddings[token][embedding_key] = utteranceDict[u][embedding_key]
                            else:
                                lexiconEmbeddings[token][embedding_key] += utteranceDict[u][embedding_key]

                except:
                    pass
        embedding_keys = [key for key in list(trainingUtterances[list(trainingUtterances.keys())[0]].keys()) if key.startswith('embed')]

        # Token embeddings: average and normalize by the average embedding.
        averageEmbeddings = {}
        for key in embedding_keys:
            averageEmbeddings[key] = np.mean([trainingUtterances[u][key] for u in trainingUtterances])
        for token in list(lexiconEmbeddings.keys()):
            if embedding_keys[0] not in lexiconEmbeddings[token]:
                del lexiconEmbeddings[token]
            else:
                for key in embedding_keys:
                    lexiconEmbeddings[token][key] = (lexiconEmbeddings[token][key] / lexiconEmbeddings[token]['counts']) # - averageEmbeddings[key]
                

        # Analogy embeddings: randomly select analogies, but only keep around the best.
        random_seed, num_analogies = 0, 100
        random_gen = random.Random(random_seed)
        analogies = [tuple(random_gen.sample(trainingUtterances.keys(), 2)) for _ in range(num_analogies)]
        analogyDict = {"%s : %s" % (a, b) : {'frontiers' : 0} for a, b in analogies} # Add dummy 'frontiers' variable for consistency.
        for analogy in analogies:
            a, b = analogy
            for embedding_key in embedding_keys:
                analogyDict["%s : %s" % (a, b)][embedding_key] = trainingUtterances[a][embedding_key] - trainingUtterances[b][embedding_key]

        # Manual analogy embeddings: evaluate handwritten analogies. (e.g. 'go to heart' : 'go to left of heart' :: 'go to circle' : ??) 
        manualAnalogyDict = {"%s : %s :: %s : " % (a, b, c) : {'frontiers' : 0} for (a, b, c) in ANALOGIES} # Add dummy 'frontiers' variable for consistency.
        for analogy in ANALOGIES:
            a, b, c = analogy
            for embedding_key in embedding_keys:
                manualAnalogyDict["%s : %s :: %s : " % (a, b, c)][embedding_key] = (manualAnalogyUtterances[b][embedding_key] - manualAnalogyUtterances[a][embedding_key]) + manualAnalogyUtterances[c][embedding_key]

        # Manual task minus other task embeddings.
        task_subtractions = []
        for directionToken in ['left', 'right', 'above', 'below']:
            for obj1 in obj_strs:
                dir_string = directionToken if directionToken not in ['left', 'right'] else directionToken + " of"
                task_subtractions.append(('reach %s the %s' % (dir_string, obj1), ('reach the %s' % obj1)))
                task_subtractions.append(('reach %s the %s' % (dir_string, obj1), ('reach %s' % directionToken)))
                #word_subtractions.append(('%s the %s' % (dir_string, obj1), directionToken))
                #word_subtractions.append(('go %s the %s' % (dir_string, obj1), directionToken))
        taskSubtractionDict = {"%s - %s" % (a, b) : {'frontiers' : 0} for (a, b) in task_subtractions}
        for taskSubtraction in task_subtractions:
            a, b = taskSubtraction
            for embedding_key in embedding_keys:
                taskSubtractionDict["%s - %s" % (a, b)][embedding_key] = manualAnalogyUtterances[a][embedding_key]  - manualAnalogyUtterances[b][embedding_key]

        # Manual task minus lexicon embeddings.
        word_subtractions = []
        for directionToken in ['left', 'right', 'above', 'below']:
            for obj1 in obj_strs:
                dir_string = directionToken if directionToken not in ['left', 'right'] else directionToken + " of"
                word_subtractions.append(('reach %s the %s' % (dir_string, obj1), directionToken))
                #word_subtractions.append(('%s the %s' % (dir_string, obj1), directionToken))
                #word_subtractions.append(('go %s the %s' % (dir_string, obj1), directionToken))
                word_subtractions.append(('reach %s the %s' % (dir_string, obj1), obj1))
                word_subtractions.append(('reach two %s the %s' % (dir_string, obj1), 'two'))
                word_subtractions.append(('reach two %s the %s' % (dir_string, obj1), directionToken))
                word_subtractions.append(('reach two %s the %s' % (dir_string, obj1), obj1))

        wordSubtractionDict = {"%s - %s" % (a, b) : {'frontiers' : 0} for (a, b) in word_subtractions}
        for wordSubtraction in word_subtractions:
            a, b = wordSubtraction
            for embedding_key in embedding_keys:
                wordSubtractionDict["%s - %s" % (a, b)][embedding_key] = manualAnalogyUtterances[a][embedding_key]  - directLexiconEmbeddings[b][embedding_key]

        
        # Calculate kNN utterances.
        for embedding_key in embedding_keys:
                kNN(trainingUtterances, trainingUtterances, 'train', embedding_key)
                kNN(unsolvedUtterances, trainingUtterances, 'train', embedding_key)
                #kNN(analogyDict, trainingUtterances, 'train', embedding_key)
                #kNN(analogyDict, analogyDict, 'analogy', embedding_key)
                kNN(manualAnalogyDict, trainingUtterances, 'train', embedding_key, closest_n=20)
                kNN(lexiconEmbeddings, lexiconEmbeddings, 'train', embedding_key, closest_n=20)
                kNN(taskSubtractionDict, trainingUtterances, 'train', embedding_key, closest_n=20)
                kNN(wordSubtractionDict, trainingUtterances, 'train', embedding_key, closest_n=20)

        for utteranceDict in trainingUtterances, unsolvedUtterances, analogyDict, lexiconEmbeddings, directLexiconEmbeddings, manualAnalogyDict, taskSubtractionDict, wordSubtractionDict: # removed: lexiconEmbeddings
            for utterance in utteranceDict:
                # Check that we have anything to print first:
                if all([not key.startswith('kNN') for key in utteranceDict[utterance].keys()]):
                    continue
                else:
                    if 'frontiers' in utteranceDict[utterance]:
                        print("%s : %d frontiers" % (utterance.upper(), utteranceDict[utterance]['frontiers']))
                    else:
                        print("%s : %d frontiers" % (utterance.upper(), utteranceDict[utterance]['counts']))
                    for feature_key in list(utteranceDict[utterance].keys()):
                        if feature_key.startswith('kNN') and feature_key in utteranceDict[utterance]:
                            print(feature_key)
                            labels, distances, programs = utteranceDict[utterance][feature_key]
                            print("\t\n".join(labels))
                            print(programs[0])
                    print("\n")

        for key in embedding_keys:
            plotTSNE(key, lexiconEmbeddings, key)
            plotTSNE(key, directLexiconEmbeddings, key)


        assert False
        #### Task-specific TSNE plots.

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




