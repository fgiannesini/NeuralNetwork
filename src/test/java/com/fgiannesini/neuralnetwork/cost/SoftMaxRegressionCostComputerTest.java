package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class SoftMaxRegressionCostComputerTest {
    @Test
    void compute_cost_on_vector() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.SOFT_MAX)
                .build();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        double cost = new SoftMaxRegressionCostComputer(outputComputer)
                .compute(new DoubleMatrix(3, 1, 1, 1, 1), new DoubleMatrix(2, 1, 1, 1));
        Assertions.assertEquals(1.3862, cost, 0.01);
    }

    @Test
    void compute_cost_on_matrix() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.SOFT_MAX)
                .build();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        double cost = new SoftMaxRegressionCostComputer(outputComputer)
                .compute(new DoubleMatrix(3, 2, 1, 1, 1, 1, 1, 1),
                        new DoubleMatrix(2, 2, 1, 1, 1, 1)
                );
        Assertions.assertEquals(1.3862, cost, 0.01);
    }

}