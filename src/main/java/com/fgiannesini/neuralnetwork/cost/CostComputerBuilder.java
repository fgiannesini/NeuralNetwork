package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class CostComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;
    private CostType costType;
    private Double l2RegularizationCoeff;

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

    public CostComputerBuilder withL2Regularization(double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
        return this;
    }

    public CostComputer build() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        CostComputer costComputer;
        switch (costType) {
            case LOGISTIC_REGRESSION:
                costComputer = new LogisticRegressionCostComputer(neuralNetworkModel);
                break;
            case LINEAR_REGRESSION:
                costComputer = new LinearRegressionCostComputer(neuralNetworkModel);
                break;
            default:
                throw new IllegalArgumentException(costType + " instantiation is not implemented");
        }

        if (l2RegularizationCoeff != null) {
            costComputer = new CostComputerWithL2LinearRegression(neuralNetworkModel, costComputer, l2RegularizationCoeff);
        }
        return costComputer;
    }
}
