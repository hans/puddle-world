"""
Utilities to explore the EC recognition model from loaded checkpoints.
"""

####
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../ec/")
####


if __name__ == "__main__":
	    import sys

        import argparse

        parser = argparse.ArgumentParser(description = "")
        parser.add_argument("--checkpoints",nargs='+')
        parser.add_argument ("--experimentNames", nargs='+', type=str, default=None)
        parser.add_argument("--metricsToPlot", nargs='+',type=str)
        parser.add_argument("--times", type=str, default='recognitionBestTimes')
        parser.add_argument("--exportTaskTimes", type=bool)
        parser.add_argument("--outlierThreshold", type=float, default=None)

        #TSNE 
        parser.add_argument("--metricsToCluster", nargs='+', type=str, default=None)
        parser.add_argument("--tsneLearningRate", type=float, default=250.0)
        parser.add_argument("--tsnePerplexity", type=float, default=30.0)
        parser.add_argument("--labelWithImages", type=bool, default=None)
        parser.add_argument("--labelsAndImages", default=False, action="store_true")
        parser.add_argument('--printExamples', type=str, default=None)
        parser.add_argument('--applySoftmax',  default=False, action="store_true")