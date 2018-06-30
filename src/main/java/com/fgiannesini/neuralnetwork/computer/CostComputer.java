package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class CostComputer {

    private final OutputComputer outputComputer;

    public CostComputer(NeuralNetworkModel neuralNetworkModel) {
        outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .build();
    }

    public float compute(float[] input, float[] output) {
        return compute(new FloatMatrix(input), new FloatMatrix(output));
    }

    public float compute(float[][] input, float[][] output) {
        return compute(new FloatMatrix(input).transpose(), new FloatMatrix(output).transpose());
    }

    public float compute(FloatMatrix input, FloatMatrix output) {
        float inputCount = input.columns;
        FloatMatrix computedOutput = outputComputer.computeOutput(input);
        FloatMatrix firstPart = MatrixFunctions.log(computedOutput).muli(output);
        FloatMatrix secondPart = MatrixFunctions.log(computedOutput.sub(1).muli(-1)).muli(output.sub(1).muli(-1));
        return firstPart.addi(secondPart).sum() / inputCount;
    }
}
