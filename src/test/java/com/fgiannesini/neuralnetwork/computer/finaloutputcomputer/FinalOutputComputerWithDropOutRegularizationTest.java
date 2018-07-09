package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

class FinalOutputComputerWithDropOutRegularizationTest {

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
        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0, 1.2, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );
        double[] output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(inputData);
        Assertions.assertArrayEquals(new double[]{10.6, 10.6}, output);
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

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        double[] output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(inputData);
        Assertions.assertArrayEquals(new double[]{18.46, 18.46}, output);
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

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        double[][] output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(inputData);
        double[][] expected = {{18.46, 18.46}, {28.18, 28.18}};
        Assertions.assertEquals(expected.length, output.length);
        IntStream.range(0, expected.length).forEach(i -> Assertions.assertArrayEquals(expected[i], output[i], 0.0001));
    }
}