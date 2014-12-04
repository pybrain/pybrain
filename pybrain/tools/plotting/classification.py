"""
matplotlib helpers for ClassificationDataSet and classifiers in general.
"""
__author__ = 'Werner Beroux <werner@beroux.com>'

import numpy as np
import matplotlib.pyplot as plt

class ClassificationDataSetPlot(object):
    @staticmethod
    def plot_module_classification_sequence_performance(module, dataset, sequence_index, bounds=(0, 1)):
        """Plot all outputs and fill the value of the output of the correct category.

        The grapth of a good classifier should be like all white, with all other
        values very low. A graph with lot of black is a bad sign.

        :param module: The module/network to plot.
        :type module: pybrain.structure.modules.module.Module
        :param dataset: Training dataset used as inputs and expected outputs.
        :type dataset: SequenceClassificationDataSet
        :param sequence_index: Sequence index to plot in the dataset.
        :type sequence_index: int
        :param bounds: Outputs lower and upper bound.
        :type bounds: list
        """
        outputs = []
        valid_output = []
        module.reset()
        for sample in dataset.getSequenceIterator(sequence_index):
            out = module.activate(sample[0])
            outputs.append(out)
            valid_output.append(out[sample[1].argmax()])
        plt.fill_between(list(range(len(valid_output))), 1, valid_output, facecolor='k', alpha=0.8)
        plt.plot(outputs, linewidth=4, alpha=0.7)
        plt.yticks(bounds)

    @staticmethod
    def plot_module_classification_dataset_performance(module, dataset, cols=4, bounds=(0, 1)):
        """Do a plot_module_classification_sequence_performance() for all sequences in the dataset.
        :param module: The module/network to plot.
        :type module: pybrain.structure.modules.module.Module
        :param dataset: Training dataset used as inputs and expected outputs.
        :type dataset: SequenceClassificationDataSet
        :param bounds: Outputs lower and upper bound.
        :type bounds: list
        """
        # Outputs and detected category error for each sequence.
        for i in range(dataset.getNumSequences()):
            plt.subplot(ceil(dataset.getNumSequences() / float(cols)), cols, i)
            ClassificationDataSetPlot.plot_module_classification_sequence_performance(module, dataset, i, bounds)

    @staticmethod
    def punchcard_module_classification_performance(module, dataset, s=800):
        """Punshcard-like clasification performances.__add__(

        Actual dataset target vs. estimated target by the module.
        The graph of a good classfier module should a have no red dots visible:
        - Red Dots: Target (only visible if the black dot doesn't cover it).
        - Green Dots: Estimated classes confidences (size = outputs means).
        - Black Dots: Single winnter-takes-all estimated target.

        :param module: An object that has at least reset() and activate() methods.
        :param dataset: A classification dataset. It should, for any given sequence, have a constant target.
        :type dataset: ClassificationDataSet
        """
        # TODO: Could also show the variation for each dot
        #       (e.g., vertical errorbar of 2*stddev).
        # TODO: Could keep together all sequences of a given class and somehow
        #       arrange them closer togther. Could then aggregate them and
        #       include horizontal errorbar.

        def calculate_module_output_mean(module, inputs):
            """Returns the mean of the module's outputs for a given input list."""
            outputs = np.zeros(module.outdim)
            module.reset()
            for inpt in inputs:
                outputs += module.activate(inpt)
            return outputs / len(inputs)

        num_sequences = dataset.getNumSequences()
        actual = []
        expected = []
        confidence_x = []
        confidence_s = []
        correct = 0

        for seq_i in range(num_sequences):
            seq = dataset.getSequence(seq_i)
            outputs_mean = calculate_module_output_mean(module, seq[0])
            actual.append(np.argmax(outputs_mean))
            confidence_s.append(np.array(outputs_mean))
            confidence_x.append(np.ones(module.outdim) * seq_i)
            # FIXME: np.argmax(seq[1]) == dataset.getSequenceClass(seq_i) is bugged for split SequenceClassificationDataSet.
            expected.append(np.argmax(seq[1]))
            if actual[-1] == expected[-1]:
                correct += 1

        plt.title('{}% Correct Classification (red dots mean bad classification)'.format(correct * 100 / num_sequences))
        plt.xlabel('Sequence')
        plt.ylabel('Class')
        plt.scatter(list(range(num_sequences)), expected, s=s, c='r', linewidths=0)
        plt.scatter(list(range(num_sequences)), actual, s=s, c='k')
        plt.scatter(confidence_x, list(range(module.outdim)) * num_sequences, s=s*np.array(confidence_s), c='g', linewidths=0, alpha=0.66)
        plt.yticks(list(range(dataset.nClasses)), dataset.class_labels)
