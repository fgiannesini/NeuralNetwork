package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.function.Function;

public class GradientDescentWithDerivationContainer {

    private final LayerTypeData input;
    private final LayerTypeData y;
    private final NeuralNetworkModel neuralNetworkModel;
    private final CostType costType;
    private final Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> costComputerProcessLauncher;

    public GradientDescentWithDerivationContainer(LayerTypeData inputData, LayerTypeData outputData, NeuralNetworkModel neuralNetworkModel, CostType costType, Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> costComputerProcessLauncher) {
        this.input = inputData;
        this.y = outputData;
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.costComputerProcessLauncher = costComputerProcessLauncher;
    }

    public LayerTypeData getInput() {
        return input;
    }

    public LayerTypeData getY() {
        return y;
    }

    public NeuralNetworkModel getNeuralNetworkModel() {
        return neuralNetworkModel;
    }

    public CostType getCostType() {
        return costType;
    }

    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerProcessLauncher() {
        return costComputerProcessLauncher;
    }
}
