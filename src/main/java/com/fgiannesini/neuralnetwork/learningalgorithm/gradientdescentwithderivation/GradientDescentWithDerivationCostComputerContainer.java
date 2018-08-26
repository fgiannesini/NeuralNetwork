package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public class GradientDescentWithDerivationCostComputerContainer {
    private final NeuralNetworkModel<Layer> neuralNetworkModel;
    private final CostType costType;

    public GradientDescentWithDerivationCostComputerContainer(NeuralNetworkModel<Layer> neuralNetworkModel, CostType costType) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
    }

    public NeuralNetworkModel<Layer> getNeuralNetworkModel() {
        return neuralNetworkModel;
    }

    public CostType getCostType() {
        return costType;
    }
}
