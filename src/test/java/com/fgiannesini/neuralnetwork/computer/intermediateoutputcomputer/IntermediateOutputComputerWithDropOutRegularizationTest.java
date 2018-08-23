package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

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

class IntermediateOutputComputerWithDropOutRegularizationTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildWeightBiasModel();

        double[] inputData = new double[3];
        Arrays.fill(inputData, 1);
        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0, 1.2, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );
        List<double[]> output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        Assertions.assertEquals(3, output.size());
        Assertions.assertArrayEquals(new double[]{1, 1, 1}, output.get(0));
        Assertions.assertArrayEquals(new double[]{4.8, 0, 4.8, 0}, output.get(1));
        Assertions.assertArrayEquals(new double[]{10.6, 10.6}, output.get(2));
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
                .buildWeightBiasModel();

        double[] inputData = new double[3];
        Arrays.fill(inputData, 1);

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        List<double[]> output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);
        Assertions.assertEquals(5, output.size());
        Assertions.assertArrayEquals(new double[]{1, 1, 1}, output.get(0));
        Assertions.assertArrayEquals(new double[]{4.8, 0}, output.get(1));
        Assertions.assertArrayEquals(new double[]{0, 8.7}, output.get(2));
        Assertions.assertArrayEquals(new double[]{17.46, 0}, output.get(3));
        Assertions.assertArrayEquals(new double[]{18.46, 18.46}, output.get(4));
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
                .buildWeightBiasModel();

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

        List<double[][]> outputs = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);
        check2DTabs(outputs.get(0), new double[][]{{1, 1, 1}, {2, 2, 2}});
        check2DTabs(outputs.get(1), new double[][]{{4.8, 0}, {8.4, 0}});
        check2DTabs(outputs.get(2), new double[][]{{0, 8.7}, {0, 14.1}});
        check2DTabs(outputs.get(3), new double[][]{{17.46, 0}, {27.18, 0}});
        check2DTabs(outputs.get(4), new double[][]{{18.46, 18.46}, {28.18, 28.18}});
    }

    private void check2DTabs(double[][] output, double[][] expected) {
        Assertions.assertEquals(expected.length, output.length);
        IntStream.range(0, expected.length).forEach(i -> Assertions.assertArrayEquals(expected[i], output[i], 0.0001));
    }
}