package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class GradientLayerProviderBuilderTest {

    @Test
    void results_are_not_present() {
        NeuralNetworkModel neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addWeightBiasLayer(2, ActivationFunctionType.RELU)
                        .buildNeuralNetworkModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void intermediate_results_are_not_present_on_batch_norm_type() {
        NeuralNetworkModel neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addBatchNormLayer(2, ActivationFunctionType.RELU)
                        .buildNeuralNetworkModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void results_and_intermediate_results_are_not_present_on_weight_bias_type() {
        NeuralNetworkModel neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addWeightBiasLayer(2, ActivationFunctionType.RELU)
                        .buildNeuralNetworkModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void instanciate_batch_norm_provider() {
        NeuralNetworkModel neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addBatchNormLayer(2, ActivationFunctionType.RELU)
                        .buildNeuralNetworkModel();

        MeanDeviationProvider meanDeviationProvider = new MeanDeviationProvider();
        List<IntermediateOutputResult> intermediateOutputResultList = Arrays.asList(
                new IntermediateOutputResult(new BatchNormData(DoubleMatrix.ones(2), meanDeviationProvider)),
                new IntermediateOutputResult(new BatchNormData(DoubleMatrix.ones(2), meanDeviationProvider))
        );
        List<GradientLayerProvider> build = GradientLayerProviderBuilder.init()
                .withModel(neuralNetwork)
                .withIntermediateResults(intermediateOutputResultList)
                .build();
        build.forEach(i -> Assertions.assertTrue(i instanceof GradientBatchNormLayerProvider));
    }

    @Test
    void instanciate_weight_bias_provider() {
        NeuralNetworkModel neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addWeightBiasLayer(2, ActivationFunctionType.RELU)
                        .buildNeuralNetworkModel();

        List<IntermediateOutputResult> intermediateOutputResultList = Arrays.asList(
                new IntermediateOutputResult(new WeightBiasData(DoubleMatrix.ones(2))),
                new IntermediateOutputResult(new WeightBiasData(DoubleMatrix.ones(2)))
        );
        List<GradientLayerProvider> build = GradientLayerProviderBuilder.init()
                .withModel(neuralNetwork)
                .withIntermediateResults(intermediateOutputResultList)
                .build();
        build.forEach(i -> Assertions.assertTrue(i instanceof GradientWeightBiasLayerProvider));
    }
}