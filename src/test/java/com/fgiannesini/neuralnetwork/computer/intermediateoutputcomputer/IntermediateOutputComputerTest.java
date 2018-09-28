package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class IntermediateOutputComputerTest {

    @Test
    void compute_one_dimension_output_with_one_hidden_weight_bias_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();


        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getInput(), ((WeightBiasData) output.get(0).getResult()).getInput());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 4, 4, 4, 4), ((WeightBiasData) output.get(1).getResult()).getInput());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 17, 17), ((WeightBiasData) output.get(2).getResult()).getInput());
    }

    @Test
    void compute_one_dimension_output_with_one_hidden_batch_norm_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(4, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();


        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getInput(), ((WeightBiasData) output.get(0).getResult()).getInput());
        Assertions.assertNull(output.get(0).getMeanDeviation());
        Assertions.assertNull(output.get(0).getAfterMeanApplicationResult());
        Assertions.assertNull(output.get(0).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 1, 1, 1, 1), ((WeightBiasData) output.get(1).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(4, 1).muli(3), output.get(1).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 1, 1), ((WeightBiasData) output.get(2).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(4), output.get(2).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getBeforeNormalisationResult());
    }

    @Test
    void compute_one_dimension_output_with_three_hidden_weight_bias_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 1, 1, 1, 1));

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getInput(), ((WeightBiasData) output.get(0).getResult()).getInput());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 4, 4), ((WeightBiasData) output.get(1).getResult()).getInput());
        Assertions.assertNull(output.get(1).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 9, 9), ((WeightBiasData) output.get(2).getResult()).getInput());
        Assertions.assertNull(output.get(2).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 19, 19), ((WeightBiasData) output.get(3).getResult()).getInput());
        Assertions.assertNull(output.get(3).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 39, 39), ((WeightBiasData) output.get(4).getResult()).getInput());
        Assertions.assertNull(output.get(4).getMeanDeviation());
    }

    @Test
    void compute_two_dimension_output_with_three_hidden_weight_bias_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .addWeightBiasLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        WeightBiasData inputData = new WeightBiasData(new DoubleMatrix(3, 2, 1, 1, 1, 2, 2, 2));

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getInput(), ((WeightBiasData) output.get(0).getResult()).getInput());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 4, 4, 7, 7), ((WeightBiasData) output.get(1).getResult()).getInput());
        Assertions.assertNull(output.get(1).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 9, 9, 15, 15), ((WeightBiasData) output.get(2).getResult()).getInput());
        Assertions.assertNull(output.get(2).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 19, 19, 31, 31), ((WeightBiasData) output.get(3).getResult()).getInput());
        Assertions.assertNull(output.get(3).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 39, 39, 63, 63), ((WeightBiasData) output.get(4).getResult()).getInput());
        Assertions.assertNull(output.get(4).getMeanDeviation());
    }

    @Test
    void compute_two_dimension_output_with_three_hidden_batch_norm_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addBatchNormLayer(2, ActivationFunctionType.NONE)
                .addBatchNormLayer(2, ActivationFunctionType.NONE)
                .addBatchNormLayer(2, ActivationFunctionType.NONE)
                .addBatchNormLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();


        BatchNormData inputData = new BatchNormData(new DoubleMatrix(3, 2, 1, 1, 1, 2, 2, 2), new MeanDeviationProvider());

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getInput(), ((BatchNormData) output.get(0).getResult()).getInput());
        Assertions.assertNull(output.get(0).getMeanDeviation());
        Assertions.assertNull(output.get(0).getBeforeNormalisationResult());
        Assertions.assertNull(output.get(0).getAfterMeanApplicationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(1).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(4.5), output.get(1).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(1.5), output.get(1).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1.5, -1.5, 1.5, 1.5), output.get(1).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(1).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(2).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(2).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(2).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(2).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(2).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(3).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(3).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(3).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(3).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(3).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(4).getResult()).getInput());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(4).getMeanDeviation().getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), output.get(4).getMeanDeviation().getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(4).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(4).getBeforeNormalisationResult());
    }
}