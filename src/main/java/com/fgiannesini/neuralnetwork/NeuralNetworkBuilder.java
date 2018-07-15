package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.normalizer.INormalizer;
import com.fgiannesini.neuralnetwork.normalizer.NormalizerType;

public class NeuralNetworkBuilder {

    private LearningAlgorithm learningAlgorithm;
    private INormalizer normalizer;

    private NeuralNetworkBuilder() {
        normalizer = NormalizerType.NONE.get();
    }

    public static NeuralNetworkBuilder init() {
        return new NeuralNetworkBuilder();
    }

    public NeuralNetworkBuilder withLearningAlgorithm(LearningAlgorithm learningAlgorithm) {
        this.learningAlgorithm = learningAlgorithm;
        return this;
    }

    public NeuralNetworkBuilder withNormalizer(INormalizer normalizer) {
        this.normalizer = normalizer;
        return this;
    }

    public NeuralNetwork build() {
        checkInputs();
        return new NeuralNetwork(learningAlgorithm, normalizer);
    }

    private void checkInputs() {
        if (learningAlgorithm == null) {
            throw new IllegalArgumentException("A learning algorithm should be set");
        }
    }
}
