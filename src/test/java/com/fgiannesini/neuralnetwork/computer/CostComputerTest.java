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
                .inputSize(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .outputSize(2)
                .build();
        float cost = new CostComputer(neuralNetworkModel)
                .compute(new float[]{1f, 1f, 1f}, new float[]{12, 12});
        Assertions.assertEquals(0, cost);
    }

    @Test
    void compute_cost_on_matrix_cost_is_zero() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .inputSize(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .outputSize(2)
                .build();
        float cost = new CostComputer(neuralNetworkModel)
                .compute(new float[][]{
                        {1f, 1f, 1f},
                        {1f, 1f, 1f}
                }, new float[][]{
                        {12, 12},
                        {12, 12}
                });
        Assertions.assertEquals(0, cost);
    }

}