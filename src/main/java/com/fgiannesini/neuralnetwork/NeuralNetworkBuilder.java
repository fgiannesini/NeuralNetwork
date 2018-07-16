package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithmBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithmType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.INormalizer;
import com.fgiannesini.neuralnetwork.normalizer.NormalizerType;

public class NeuralNetworkBuilder {

    private INormalizer normalizer;
    private NeuralNetworkModel neuralNetworkModel;
    private LearningAlgorithmType learningAlgorithmType;

    private NeuralNetworkBuilder() {
        normalizer = NormalizerType.NONE.get();
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
    }

    public static NeuralNetworkBuilder init() {
        return new NeuralNetworkBuilder();
    }

    public NeuralNetworkBuilder withNormalizer(INormalizer normalizer) {
        this.normalizer = normalizer;
        return this;
    }

    public NeuralNetworkBuilder withNeuralNetworkModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public NeuralNetworkBuilder withLearningAlgorithmType(LearningAlgorithmType learningAlgorithmType) {
        this.learningAlgorithmType = learningAlgorithmType;
        return this;
    }

    public NeuralNetwork build() {
        checkInputs();
        LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                .withModel(neuralNetworkModel)
                .build();
        return new NeuralNetwork(learningAlgorithm, normalizer);
    }

    private void checkInputs() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("A neuralNetworkModel type should be set");
        }
    }
}
