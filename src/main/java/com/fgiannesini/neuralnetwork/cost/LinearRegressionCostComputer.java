package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.FinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class LinearRegressionCostComputer implements CostComputer {

    private final FinalOutputComputer outputComputer;

    public LinearRegressionCostComputer(NeuralNetworkModel neuralNetworkModel) {
        outputComputer = OutputComputerBuilder.init().withModel(neuralNetworkModel).buildFinalOutputComputer();
    }

    @Override
    public double compute(DoubleMatrix input, DoubleMatrix output) {
        DoubleMatrix computedOutput = outputComputer.compute(input);
        double inputCount = input.getColumns();
        return computedOutput.squaredDistance(output) / (inputCount * 2d);
    }
}
