package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CostComputerTest {

    @Test
    void compute_cost_on_vector_result_is_zero() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .output(2)
                .build();
        double cost = new CostComputer(neuralNetworkModel)
                .compute(new double[]{1f, 1f, 1f}, new double[]{12, 12});
        Assertions.assertEquals(0, cost);
    }

    @Test
    void compute_cost_on_matrix_cost_is_zero() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .output(2)
                .build();
        double cost = new CostComputer(neuralNetworkModel)
                .compute(new double[][]{
                        {1f, 1f, 1f},
                        {1f, 1f, 1f}
                }, new double[][]{
                        {12, 12},
                        {12, 12}
                });
        Assertions.assertEquals(0, cost);
    }

}