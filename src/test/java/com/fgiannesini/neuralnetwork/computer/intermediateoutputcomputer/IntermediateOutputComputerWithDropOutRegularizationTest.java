package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class IntermediateOutputComputerWithDropOutRegularizationTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));
        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0, 1.2, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );
        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        Assertions.assertEquals(3, output.size());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 1, 1, 1, 1), ((WeightBiasData) output.get(0).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 4.8, 0, 4.8, 0), ((WeightBiasData) output.get(1).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 10.6, 10.6), ((WeightBiasData) output.get(2).getResult()).getData());
    }

    @Test
    void compute_one_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        Assertions.assertEquals(5, output.size());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 1, 1, 1, 1), ((WeightBiasData) output.get(0).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 4.8, 0), ((WeightBiasData) output.get(1).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 0, 8.7), ((WeightBiasData) output.get(2).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 17.46, 0), ((WeightBiasData) output.get(3).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 18.46, 18.46), ((WeightBiasData) output.get(4).getResult()).getData());
    }

    @Test
    void compute_two_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        LayerTypeData inputData = new WeightBiasData(new DoubleMatrix(3, 2, 1, 1, 1, 2, 2, 2));

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer()
                .compute(inputData);
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(3, 2, 1, 1, 1, 2, 2, 2), ((WeightBiasData) output.get(0).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 4.8, 0, 8.4, 0), ((WeightBiasData) output.get(1).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 8.7, 0, 14.1), ((WeightBiasData) output.get(2).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 17.46, 0, 27.18, 0), ((WeightBiasData) output.get(3).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 18.46, 18.46, 28.18, 28.18), ((WeightBiasData) output.get(4).getResult()).getData());
    }
}