package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;

public class GradientDescentCorrectionsContainer {
    private NeuralNetworkModel correctedNeuralNetworkModel;
    private List<GradientDescentCorrection> gradientDescentCorrections;
    private int inputCount;
    private double learningRate;

    public GradientDescentCorrectionsContainer(NeuralNetworkModel correctedNeuralNetworkModel, List<GradientDescentCorrection> gradientDescentCorrections, int inputCount, double learningRate) {
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
