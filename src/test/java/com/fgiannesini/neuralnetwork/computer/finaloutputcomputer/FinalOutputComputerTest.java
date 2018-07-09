package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class FinalOutputComputerTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        double[] inputData = new double[3];
        Arrays.fill(inputData, 1);

        double[] output = OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        Assertions.assertArrayEquals(new double[]{17, 17}, output);
    }

    @Test
    void compute_one_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        double[] inputData = new double[3];
        Arrays.fill(inputData, 1);

        double[] output = OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        Assertions.assertArrayEquals(new double[]{39, 39}, output);
    }

    @Test
    void compute_two_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        double[][] inputData = {
                {1, 1, 1},
                {2, 2, 2}
        };

        double[][] output = OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        double[][] expected = {{39, 39}, {63, 63}};
        Assertions.assertArrayEquals(expected, output);
    }
}