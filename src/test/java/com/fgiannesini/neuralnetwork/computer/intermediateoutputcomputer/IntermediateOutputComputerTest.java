package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class IntermediateOutputComputerTest {

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

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<double[]> output = outputComputer
                .compute(inputData);
        Assertions.assertArrayEquals(inputData, output.get(0));
        Assertions.assertArrayEquals(new double[]{4, 4, 4, 4}, output.get(1));
        Assertions.assertArrayEquals(new double[]{17, 17}, output.get(2));
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

        List<double[]> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);
        Assertions.assertArrayEquals(inputData, output.get(0));
        Assertions.assertArrayEquals(new double[]{4, 4}, output.get(1));
        Assertions.assertArrayEquals(new double[]{9, 9}, output.get(2));
        Assertions.assertArrayEquals(new double[]{19, 19}, output.get(3));
        Assertions.assertArrayEquals(new double[]{39, 39}, output.get(4));
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

        List<double[][]> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        Assertions.assertArrayEquals(inputData, output.get(0));
        Assertions.assertArrayEquals(new double[][]{{4, 4}, {7, 7}}, output.get(1));
        Assertions.assertArrayEquals(new double[][]{{9, 9}, {15, 15}}, output.get(2));
        Assertions.assertArrayEquals(new double[][]{{19, 19}, {31, 31}}, output.get(3));
        Assertions.assertArrayEquals(new double[][]{{39, 39}, {63, 63}}, output.get(4));
    }
}