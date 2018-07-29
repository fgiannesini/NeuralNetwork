package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class GradientDescentWithDerivationAndL2Regularization extends GradientDescentWithDerivation {

    private final CostType costType;
    private final double regularizationCoeff;

    public GradientDescentWithDerivationAndL2Regularization(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, double regularizationCoeff) {
        super(neuralNetworkModel, costType, learningRate);
        this.costType = costType;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    protected CostComputer buildCostComputer(NeuralNetworkModel modifiedNeuralNetworkModel) {
        return CostComputerBuilder.init()
                .withNeuralNetworkModel(modifiedNeuralNetworkModel)
                .withType(costType)
                .withL2Regularization(regularizationCoeff)
                .build();
    }
}
