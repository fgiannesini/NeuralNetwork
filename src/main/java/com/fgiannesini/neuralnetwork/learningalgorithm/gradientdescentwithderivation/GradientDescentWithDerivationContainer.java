package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class GradientDescentWithDerivationContainer {

    private final DoubleMatrix input;
    private final DoubleMatrix y;
    private final NeuralNetworkModel<Layer> neuralNetworkModel;
    private final CostType costType;
    private final Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> costComputerProcessLauncher;

    public GradientDescentWithDerivationContainer(DoubleMatrix input, DoubleMatrix y, NeuralNetworkModel<Layer> neuralNetworkModel, CostType costType, Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> costComputerProcessLauncher) {
        this.input = input;
        this.y = y;
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.costComputerProcessLauncher = costComputerProcessLauncher;
    }

    public DoubleMatrix getInput() {
        return input;
    }

    public DoubleMatrix getY() {
        return y;
    }

    public NeuralNetworkModel<Layer> getNeuralNetworkModel() {
        return neuralNetworkModel;
    }

    public CostType getCostType() {
        return costType;
    }

    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerProcessLauncher() {
        return costComputerProcessLauncher;
    }
}
