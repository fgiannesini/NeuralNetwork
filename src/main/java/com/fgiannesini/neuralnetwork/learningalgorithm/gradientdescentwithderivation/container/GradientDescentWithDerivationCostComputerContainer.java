package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class GradientDescentWithDerivationCostComputerContainer {
    private final NeuralNetworkModel neuralNetworkModel;
    private final CostType costType;

    public GradientDescentWithDerivationCostComputerContainer(NeuralNetworkModel neuralNetworkModel, CostType costType) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
    }

    public NeuralNetworkModel getNeuralNetworkModel() {
        return neuralNetworkModel;
    }

    public CostType getCostType() {
        return costType;
    }
}
