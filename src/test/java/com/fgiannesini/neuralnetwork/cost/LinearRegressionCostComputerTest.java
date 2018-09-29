package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LinearRegressionCostComputerTest {

    @Test
    void compute_cost_on_vector() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .buildNeuralNetworkModel();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();

        LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 1, 1, 2));
        LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 1, 5, 6));

        double cost = new LinearRegressionCostComputer(outputComputer)
                .compute(input, output);
        Assertions.assertEquals(2.5, cost);
    }

    @Test
    void compute_cost_on_matrix() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(2)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .buildNeuralNetworkModel();
        IFinalOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();

        LayerTypeData input = new WeightBiasData(new DoubleMatrix(2, 2, 1, 2, 3, 4));
        LayerTypeData output = new WeightBiasData(new DoubleMatrix(2, 2, 5, 6, 9, 10));

        double cost = new LinearRegressionCostComputer(outputComputer)
                .compute(input, output);
        Assertions.assertEquals(2.5, cost);
    }
}