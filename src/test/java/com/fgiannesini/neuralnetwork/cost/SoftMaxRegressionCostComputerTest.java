package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
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
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.SOFT_MAX)
                .buildNeuralNetworkModel();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(2, 1, 1, 1));
        double cost = new SoftMaxRegressionCostComputer(outputComputer)
                .compute(input, output);
        Assertions.assertEquals(1.3862, cost, 0.01);
    }

    @Test
    void compute_cost_on_matrix() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.SOFT_MAX)
                .buildNeuralNetworkModel();

        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 2, 1, 1, 1, 1, 1, 1));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(2, 2, 1, 1, 1, 1));
        double cost = new SoftMaxRegressionCostComputer(outputComputer)
                .compute(input, output);
        Assertions.assertEquals(1.3862, cost, 0.01);
    }

}