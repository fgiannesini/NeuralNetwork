package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class CostComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;
    private CostType costType;

    private CostComputerBuilder() {
        costType = CostType.LINEAR_REGRESSION;
    }

    public static CostComputerBuilder init() {
        return new CostComputerBuilder();
    }

    public CostComputerBuilder withNeuralNetworkModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public CostComputerBuilder withType(CostType costType) {
        this.costType = costType;
        return this;
    }

    public CostComputer build() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        switch (costType) {
            case LOGISTIC_REGRESSION:
                return new LogisticRegressionCostComputer(neuralNetworkModel);
            case LINEAR_REGRESSION:
                return new LinearRegressionCostComputer(neuralNetworkModel);
            default:
                throw new IllegalArgumentException(costType + " instantiation is not implemented");
        }
    }
}
