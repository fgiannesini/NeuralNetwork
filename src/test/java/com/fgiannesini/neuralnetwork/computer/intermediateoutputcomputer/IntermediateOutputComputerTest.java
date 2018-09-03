package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class IntermediateOutputComputerTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_weight_bias_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildWeightBiasModel();


        DoubleMatrix inputData = new DoubleMatrix(3, 1, 1, 1, 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData, output.get(0).getResult());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 4, 4, 4, 4), output.get(1).getResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 17, 17), output.get(2).getResult());
    }

    @Test
    void compute_one_dimension_output_with_one_hidden_batch_norm_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildBatchNormModel();


        DoubleMatrix inputData = new DoubleMatrix(3, 1, 1, 1, 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData, output.get(0).getResult());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 1, 1, 1, 1), output.get(1).getResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(4, 1).muli(3), output.get(1).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getMeanDeviation().getDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 1, 1), output.get(2).getResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(4), output.get(2).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getMeanDeviation().getDeviation());
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
                .buildWeightBiasModel();

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