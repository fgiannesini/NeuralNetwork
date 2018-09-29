package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class FinalOutputComputerWithDropOutRegularizationTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        LayerTypeData input = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0, 1.2, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );
        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(input);
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 10.6, 10.6), output.getInput());
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

        LayerTypeData input = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(input);

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 18.46, 18.46), output.getInput());
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

        LayerTypeData input = new WeightBiasData(new DoubleMatrix(3, 2, 1, 1, 1, 2, 2, 2));
        List<DoubleMatrix> dropOutMatrices = Arrays.asList(
                new DoubleMatrix(new double[]{1, 1, 1}),
                new DoubleMatrix(new double[]{1.2, 0}),
                new DoubleMatrix(new double[]{0, 1.5}),
                new DoubleMatrix(new double[]{1.8, 0}),
                new DoubleMatrix(new double[]{1, 1})
        );

        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .withDropOutParameters(dropOutMatrices)
                .buildFinalOutputComputer()
                .compute(input);

        DoubleMatrixAssertions.assertMatrices(output.getInput(), new DoubleMatrix(2, 2, 18.46, 18.46, 28.18, 28.18));
    }
}