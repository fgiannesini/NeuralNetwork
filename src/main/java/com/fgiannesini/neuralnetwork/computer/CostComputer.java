package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class CostComputer {

    private final OutputComputer outputComputer;

    public CostComputer(NeuralNetworkModel neuralNetworkModel) {
        outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .build();
    }

    public double compute(double[] input, double[] output) {
        return compute(new DoubleMatrix(input), new DoubleMatrix(output));
    }

    public double compute(double[][] input, double[][] output) {
        return compute(new DoubleMatrix(input).transpose(), new DoubleMatrix(output).transpose());
    }

    public double compute(DoubleMatrix input, DoubleMatrix output) {
        double inputCount = input.columns;
        DoubleMatrix computedOutput = outputComputer.computeOutput(input);
        DoubleMatrix firstPart = MatrixFunctions.log(computedOutput).muli(output);
        DoubleMatrix secondPart = MatrixFunctions.logi(computedOutput.sub(1).muli(-1)).muli(output.sub(1).muli(-1));
        return firstPart.addi(secondPart).sum() / inputCount;
    }
}
