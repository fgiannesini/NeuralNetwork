package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

class FinalOutputComputerTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();


        LayerTypeData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 17, 17), output.getData());
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

        LayerTypeData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 39, 39), output.getData());
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
        WeightBiasData output = (WeightBiasData) OutputComputerBuilder.init()
                .withModel(model)
                .buildFinalOutputComputer()
                .compute(inputData);
        DoubleMatrixAssertions.assertMatrices(output.getData(), new DoubleMatrix(2, 2, 39, 39, 63, 63));

    }
}