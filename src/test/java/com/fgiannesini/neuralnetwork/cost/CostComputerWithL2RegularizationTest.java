package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CostComputerWithL2RegularizationTest {

    @Test
    void compute() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(3)
                .useInitializer(InitializerType.ONES)
                .addWeightBiasLayer(4, ActivationFunctionType.RELU)
                .addWeightBiasLayer(2, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();

        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3));

        CostComputerWithL2Regularization costComputerWithL2Regularization = new CostComputerWithL2Regularization(neuralNetworkModel, (inputMatrix, outputMatrix) -> 0, 0.5);
        double cost = costComputerWithL2Regularization.compute(input, output);
        Assertions.assertEquals(1.6666, cost, 0.0001);
    }
}