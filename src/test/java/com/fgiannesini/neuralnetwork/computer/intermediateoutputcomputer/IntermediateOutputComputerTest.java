package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.data.*;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.BatchNormMeanDeviation;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

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

        DoubleMatrixAssertions.assertMatrices(inputData.getData(), ((WeightBiasData) output.get(0).getResult()).getData());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 4, 4, 4, 4), ((WeightBiasData) output.get(1).getResult()).getData());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 17, 17), ((WeightBiasData) output.get(2).getResult()).getData());
    }

    @Test
    void compute_one_dimension_output_with_one_hidden_batch_norm_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addBatchNormLayer(4, ActivationFunctionType.NONE)
                .addBatchNormLayer(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .buildNeuralNetworkModel();

        BatchNormData inputData = new BatchNormData(new DoubleMatrix(3, 1, 1, 1, 1), new MeanDeviationProvider());

        List<IntermediateOutputResult> output = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer()
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getData(), ((BatchNormData) output.get(0).getResult()).getData());
        Assertions.assertNull(getBatchNormMeanDeviation(output.get(0)));
        Assertions.assertNull(output.get(0).getAfterMeanApplicationResult());
        Assertions.assertNull(output.get(0).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(4, 1, 1, 1, 1, 1), ((BatchNormData) output.get(1).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(4, 1).muli(3), getBatchNormMeanDeviation(output.get(1)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), getBatchNormMeanDeviation(output.get(1)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(4, 1), output.get(1).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 1, 1), ((BatchNormData) output.get(2).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(4), getBatchNormMeanDeviation(output.get(2)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), getBatchNormMeanDeviation(output.get(2)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.zeros(2, 1), output.get(2).getBeforeNormalisationResult());
    }

    private BatchNormMeanDeviation getBatchNormMeanDeviation(IntermediateOutputResult intermediateOutputResult) {
        return (BatchNormMeanDeviation) intermediateOutputResult.getMeanDeviation();
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

        DoubleMatrixAssertions.assertMatrices(inputData.getData(), ((WeightBiasData) output.get(0).getResult()).getData());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 4, 4), ((WeightBiasData) output.get(1).getResult()).getData());
        Assertions.assertNull(output.get(1).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 9, 9), ((WeightBiasData) output.get(2).getResult()).getData());
        Assertions.assertNull(output.get(2).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 19, 19), ((WeightBiasData) output.get(3).getResult()).getData());
        Assertions.assertNull(output.get(3).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 39, 39), ((WeightBiasData) output.get(4).getResult()).getData());
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

        DoubleMatrixAssertions.assertMatrices(inputData.getData(), ((WeightBiasData) output.get(0).getResult()).getData());
        Assertions.assertNull(output.get(0).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 4, 4, 7, 7), ((WeightBiasData) output.get(1).getResult()).getData());
        Assertions.assertNull(output.get(1).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 9, 9, 15, 15), ((WeightBiasData) output.get(2).getResult()).getData());
        Assertions.assertNull(output.get(2).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 19, 19, 31, 31), ((WeightBiasData) output.get(3).getResult()).getData());
        Assertions.assertNull(output.get(3).getMeanDeviation());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 39, 39, 63, 63), ((WeightBiasData) output.get(4).getResult()).getData());
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

        DoubleMatrixAssertions.assertMatrices(inputData.getData(), ((BatchNormData) output.get(0).getResult()).getData());
        Assertions.assertNull(getBatchNormMeanDeviation(output.get(0)));
        Assertions.assertNull(output.get(0).getBeforeNormalisationResult());
        Assertions.assertNull(output.get(0).getAfterMeanApplicationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(1).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(4.5), getBatchNormMeanDeviation(output.get(1)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(1.5), getBatchNormMeanDeviation(output.get(1)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1.5, -1.5, 1.5, 1.5), output.get(1).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(1).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(2).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(2)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(2)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(2).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(2).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(3).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(3)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(3)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(3).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(3).getBeforeNormalisationResult());

        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, 0, 0, 2, 2), ((BatchNormData) output.get(4).getResult()).getData());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(4)).getMean());
        DoubleMatrixAssertions.assertMatrices(DoubleMatrix.ones(2, 1).muli(2), getBatchNormMeanDeviation(output.get(4)).getDeviation());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -2, -2, 2, 2), output.get(4).getAfterMeanApplicationResult());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 2, -1, -1, 1, 1), output.get(4).getBeforeNormalisationResult());
    }

    @Test
    void compute_one_dimension_output_with_average_and_convolution_layer() {
        NeuralNetworkModel model = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(10, 10, 1)
                .addConvolutionLayer(3, 0, 1, 1, ActivationFunctionType.NONE)
                .addAveragePoolingLayer(3, 0, 1, ActivationFunctionType.NONE)
                .addFullyConnectedLayer(2, ActivationFunctionType.NONE)
                .buildConvolutionNetworkModel();

        ConvolutionData inputData = new ConvolutionData(Collections.singletonList(DoubleMatrix.ones(10, 10)), 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer
                .compute(inputData);

        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), ((ConvolutionData) output.get(0).getResult()).getDatas());

        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(DoubleMatrix.ones(8, 8).muli(10)), ((ConvolutionData) output.get(1).getResult()).getDatas());
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(DoubleMatrix.ones(6, 6).muli(10)), ((AveragePoolingData) output.get(2).getResult()).getDatas());
        DoubleMatrixAssertions.assertMatrices(new DoubleMatrix(2, 1, 361, 361), ((WeightBiasData) output.get(3).getResult()).getData());
    }

    @Test
    void compute_one_dimension_output_with_max_layer_and_striding() {
        NeuralNetworkModel model = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(6, 6, 1)
                .addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.NONE)
                .buildConvolutionNetworkModel();

        MaxPoolingData inputData = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(6, 6, IntStream.range(1, 37).asDoubleStream().toArray())), null, null, 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer.compute(inputData);

        MaxPoolingData firstResult = (MaxPoolingData) output.get(0).getResult();
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), firstResult.getDatas());

        MaxPoolingData secondResult = (MaxPoolingData) output.get(1).getResult();
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 8, 10, 12, 20, 22, 24, 32, 34, 36)), secondResult.getDatas());
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 1, 3, 5, 1, 3, 5, 1, 3, 5)), secondResult.getMaxRowIndexes());
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 1, 1, 1, 3, 3, 3, 5, 5, 5)), secondResult.getMaxColumnIndexes());
    }

    @Test
    void compute_one_dimension_output_with_max_layer_and_padding() {
        NeuralNetworkModel model = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3, 3, 1)
                .addMaxPoolingLayer(3, 1, 1, ActivationFunctionType.NONE)
                .buildConvolutionNetworkModel();

        MaxPoolingData inputData = new MaxPoolingData(Collections.singletonList(new DoubleMatrix(3, 3, IntStream.range(1, 10).asDoubleStream().toArray())), null, null, 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer.compute(inputData);

        MaxPoolingData firstResult = (MaxPoolingData) output.get(0).getResult();
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), firstResult.getDatas());

        MaxPoolingData secondResult = (MaxPoolingData) output.get(1).getResult();
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 5, 6, 6, 8, 9, 9, 8, 9, 9)), secondResult.getDatas());
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3)), secondResult.getMaxRowIndexes());
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3)), secondResult.getMaxColumnIndexes());
    }

    @Test
    void compute_one_dimension_output_with_average_layer_and_striding() {
        NeuralNetworkModel model = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(6, 6, 1)
                .addAveragePoolingLayer(2, 0, 2, ActivationFunctionType.NONE)
                .buildConvolutionNetworkModel();

        AveragePoolingData inputData = new AveragePoolingData(Collections.singletonList(new DoubleMatrix(6, 6, IntStream.range(1, 37).asDoubleStream().toArray())), 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer.compute(inputData);

        AveragePoolingData firstResult = (AveragePoolingData) output.get(0).getResult();
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), firstResult.getDatas());

        AveragePoolingData secondResult = (AveragePoolingData) output.get(1).getResult();
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 4.5, 6.5, 8.5, 16.5, 18.5, 20.5, 28.5, 30.5, 32.5)), secondResult.getDatas());
    }

    @Test
    void compute_one_dimension_output_with_average_layer_and_padding() {
        NeuralNetworkModel model = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .input(3, 3, 1)
                .addAveragePoolingLayer(3, 1, 1, ActivationFunctionType.NONE)
                .buildConvolutionNetworkModel();

        AveragePoolingData inputData = new AveragePoolingData(Collections.singletonList(new DoubleMatrix(3, 3, IntStream.range(1, 10).asDoubleStream().toArray())), 1);

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(model)
                .buildIntermediateOutputComputer();

        List<IntermediateOutputResult> output = outputComputer.compute(inputData);

        AveragePoolingData firstResult = (AveragePoolingData) output.get(0).getResult();
        DoubleMatrixAssertions.assertMatrices(inputData.getDatas(), firstResult.getDatas());

        AveragePoolingData secondResult = (AveragePoolingData) output.get(1).getResult();
        DoubleMatrixAssertions.assertMatrices(Collections.singletonList(new DoubleMatrix(3, 3, 1.33333, 2.33333, 1.77777, 3, 5, 3.66666, 2.66666, 4.33333, 3.11111)), secondResult.getDatas());
    }
}