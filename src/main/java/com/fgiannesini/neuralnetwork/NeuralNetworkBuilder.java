package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.cost.CostType;
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
    private double[] dropOutCoeffs;
    private Double l2RegularizationCoeff;
    private Double learningRate;
    private CostType costType;

    private NeuralNetworkBuilder() {
        normalizer = NormalizerType.NONE.get();
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
        costType = CostType.LINEAR_REGRESSION;
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

    public NeuralNetworkBuilder withDropOutRegularization(double[] dropOutCoeffs) {
        this.dropOutCoeffs = dropOutCoeffs;
        return this;
    }

    public NeuralNetworkBuilder withL2Regularization(Double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
        return this;
    }

    public NeuralNetworkBuilder withLearningRate(Double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuralNetworkBuilder withCostType(CostType costType) {
        this.costType = costType;
        return this;
    }

    public NeuralNetwork build() {
        checkInputs();
        LearningAlgorithmBuilder learningAlgorithmBuilder = LearningAlgorithmBuilder.init()
                .withModel(neuralNetworkModel)
                .withAlgorithmType(learningAlgorithmType)
                .withCostType(costType);
        if (dropOutCoeffs != null) {
            learningAlgorithmBuilder.withDropOutRegularitation(dropOutCoeffs);
        }
        if (l2RegularizationCoeff != null) {
            learningAlgorithmBuilder.withL2Regularization(l2RegularizationCoeff);
        }
        if (learningRate != null) {
            learningAlgorithmBuilder.withLearningRate(learningRate);
        }
        LearningAlgorithm learningAlgorithm = learningAlgorithmBuilder.build();
        return new NeuralNetwork(learningAlgorithm, normalizer, costType);
    }

    private void checkInputs() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("A neuralNetworkModel type should be set");
        }
    }
}
