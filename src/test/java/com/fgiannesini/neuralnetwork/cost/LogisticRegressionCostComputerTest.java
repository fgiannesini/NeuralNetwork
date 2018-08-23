package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LogisticRegressionCostComputerTest {

    @Test
    void compute_cost_on_vector_result_is_zero() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.SIGMOID)
                .buildWeightBiasModel();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        double cost = new LogisticRegressionCostComputer(outputComputer)
                .compute(new double[]{1, 1, 1}, new double[]{1, 1});
        Assertions.assertEquals(0, cost, 0.01);
    }

    @Test
    void compute_cost_on_matrix_cost_is_zero() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.SIGMOID)
                .buildWeightBiasModel();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        double cost = new LogisticRegressionCostComputer(outputComputer)
                .compute(new double[][]{
                        {1, 1, 1},
                        {1, 1, 1}
                }, new double[][]{
                        {1, 1},
                        {1, 1}
                });
        Assertions.assertEquals(0, cost, 0.01);
    }

}