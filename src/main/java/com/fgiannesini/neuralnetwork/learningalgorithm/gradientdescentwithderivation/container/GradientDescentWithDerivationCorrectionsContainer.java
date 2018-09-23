package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;

public class GradientDescentWithDerivationCorrectionsContainer {
    private final NeuralNetworkModel correctedNeuralNetworkModel;
    private final List<GradientDescentCorrection> gradientDescentCorrections;
    private final int inputCount;
    private final double learningRate;

    public GradientDescentWithDerivationCorrectionsContainer(NeuralNetworkModel correctedNeuralNetworkModel, List<GradientDescentCorrection> gradientDescentCorrections, int inputCount, double learningRate) {
        this.correctedNeuralNetworkModel = correctedNeuralNetworkModel;
        this.gradientDescentCorrections = gradientDescentCorrections;
        this.inputCount = inputCount;
        this.learningRate = learningRate;
    }

    public NeuralNetworkModel getCorrectedNeuralNetworkModel() {
        return correctedNeuralNetworkModel;
    }

    public List<GradientDescentCorrection> getGradientDescentCorrections() {
        return gradientDescentCorrections;
    }

    public int getInputCount() {
        return inputCount;
    }

    public double getLearningRate() {
        return learningRate;
    }
}
