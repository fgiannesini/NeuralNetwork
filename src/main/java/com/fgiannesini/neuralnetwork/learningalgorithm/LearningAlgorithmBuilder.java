package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class LearningAlgorithmBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private LearningAlgorithmType learningAlgorithmType;
    private double learningRate;
    private CostType costType;
    private Double l2RegularizationCoeff;

    private LearningAlgorithmBuilder() {
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
        learningRate = 0.01;
        costType = CostType.LINEAR_REGRESSION;
    }

    public static LearningAlgorithmBuilder init() {
        return new LearningAlgorithmBuilder();
    }

    public LearningAlgorithmBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public LearningAlgorithmBuilder withAlgorithmType(LearningAlgorithmType learningAlgorithmType) {
        this.learningAlgorithmType = learningAlgorithmType;
        return this;
    }

    public LearningAlgorithmBuilder withLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public LearningAlgorithmBuilder withCostType(CostType costType) {
        this.costType = costType;
        return this;
    }

    public LearningAlgorithmBuilder withL2Regularization(Double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
        return this;
    }

    public LearningAlgorithm build() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        LearningAlgorithm learningAlgorithm;
        switch (learningAlgorithmType) {
            case GRADIENT_DESCENT:
                if (l2RegularizationCoeff != null) {
                    learningAlgorithm = new GradientDescentWithL2Regularization(neuralNetworkModel, learningRate, l2RegularizationCoeff);
                } else {
                    learningAlgorithm = new GradientDescent(neuralNetworkModel, learningRate);
                }
                break;
            case GRADIENT_DESCENT_DERIVATION:
                if (l2RegularizationCoeff != null) {
                    learningAlgorithm = new GradientDescentWithDerivationAndL2Regularization(neuralNetworkModel, costType, learningRate, l2RegularizationCoeff);
                } else {
                    learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, learningRate);
                }
                break;
            default:
                throw new IllegalArgumentException(learningAlgorithmType + " instantiation is not implemented");
        }
        return learningAlgorithm;
    }

}
