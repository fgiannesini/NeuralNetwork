package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CostComputerWithL2RegularizationTest {

    @Test
    void compute() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(3)
                .useInitializer(InitializerType.ONES)
                .addLayer(4)
                .addLayer(2)
                .build();

        double[][] input = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
        double[][] output = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};

        CostComputerWithL2Regularization costComputerWithL2Regularization = new CostComputerWithL2Regularization(neuralNetworkModel, (inputMatrix, outputMatrix) -> 0, 0.5);
        double cost = costComputerWithL2Regularization.compute(input, output);
        Assertions.assertEquals(1.6666, cost, 0.0001);
    }
}