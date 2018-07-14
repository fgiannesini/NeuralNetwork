package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;

public class NeuralNetworkBuilder {

    private LearningAlgorithm learningAlgorithm;

    private NeuralNetworkBuilder() {
    }

    public static NeuralNetworkBuilder init() {
        return new NeuralNetworkBuilder();
    }

    public NeuralNetworkBuilder withLearningAlgorithm(LearningAlgorithm learningAlgorithm) {
        this.learningAlgorithm = learningAlgorithm;
        return this;
    }

    public NeuralNetwork build() {
        checkInputs();
        return new NeuralNetwork(learningAlgorithm);
    }

    private void checkInputs() {
        if (learningAlgorithm == null) {
            throw new IllegalArgumentException("A learning algorithm should be set");
        }
    }
}
