package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class LogisticRegressionCostComputer implements CostComputer {

    private final IFinalOutputComputer outputComputer;

    public LogisticRegressionCostComputer(NeuralNetworkModel neuralNetworkModel) {
        outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
    }

    @Override
    public double compute(DoubleMatrix input, DoubleMatrix output) {
        double inputCount = input.columns;
        DoubleMatrix computedOutput = outputComputer.compute(input);
        //cost = -1/m sum(y * log(^y) + (1-y) * log (1-^y))
        double result = 0;
        for (int index = 0; index < output.length; index++) {
            double outputValue = output.get(index);
            double computedOutputValue = computedOutput.get(index);
            if (outputValue != 0) {
                result += outputValue * Math.log(computedOutputValue);
            }
            if (outputValue != 1) {
                result += (1 - outputValue) * Math.log(1 - computedOutputValue);
            }
        }
        return -result / inputCount;
    }
}
