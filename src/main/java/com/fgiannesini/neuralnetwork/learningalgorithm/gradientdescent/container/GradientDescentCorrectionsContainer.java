package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;

public class GradientDescentCorrectionsContainer {
    private final NeuralNetworkModel<Layer> correctedNeuralNetworkModel;
    private final List<GradientDescentCorrection> gradientDescentCorrections;
    private final int inputCount;
    private final double learningRate;

    public GradientDescentCorrectionsContainer(NeuralNetworkModel<Layer> correctedNeuralNetworkModel, List<GradientDescentCorrection> gradientDescentCorrections, int inputCount, double learningRate) {
        this.correctedNeuralNetworkModel = correctedNeuralNetworkModel;
        this.gradientDescentCorrections = gradientDescentCorrections;
        this.inputCount = inputCount;
        this.learningRate = learningRate;
    }

    public NeuralNetworkModel<Layer> getCorrectedNeuralNetworkModel() {
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