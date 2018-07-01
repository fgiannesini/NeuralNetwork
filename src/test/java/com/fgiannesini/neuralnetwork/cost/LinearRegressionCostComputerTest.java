package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LinearRegressionCostComputerTest {

    @Test
    void compute_cost_on_vector() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();

        double[] input = {1, 2};
        double[] output = {5, 6};
        double cost = new LinearRegressionCostComputer(neuralNetworkModel)
                .compute(input, output);
        Assertions.assertEquals(2.5, cost);
    }

    @Test
    void compute_cost_on_matrix() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addLayer(2, ActivationFunctionType.NONE)
                .build();

        double[][] input = {
                {1, 2},
                {3, 4}
        };
        double[][] output = {
                {5, 6},
                {9, 10}
        };
        double cost = new LinearRegressionCostComputer(neuralNetworkModel)
                .compute(input, output);
        Assertions.assertEquals(2.5, cost);
    }
}