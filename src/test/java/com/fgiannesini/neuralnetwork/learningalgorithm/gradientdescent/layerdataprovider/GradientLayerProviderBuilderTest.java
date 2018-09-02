package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

class GradientLayerProviderBuilderTest {

    @Test
    void results_are_not_present() {
        NeuralNetworkModel<WeightBiasLayer> neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addLayer(2)
                        .buildWeightBiasModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void intermediate_results_are_not_present_on_batch_norm_type() {
        NeuralNetworkModel<BatchNormLayer> neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addLayer(2)
                        .buildBatchNormModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void results_and_intermediate_results_are_not_present_on_weight_bias_type() {
        NeuralNetworkModel<WeightBiasLayer> neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addLayer(2)
                        .buildWeightBiasModel();

        Assertions.assertThrows(IllegalArgumentException.class, () ->
                GradientLayerProviderBuilder.init()
                        .withModel(neuralNetwork)
                        .build());
    }

    @Test
    void instanciate_batch_norm_provider() {
        NeuralNetworkModel<BatchNormLayer> neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addLayer(2)
                        .buildBatchNormModel();

        GradientLayerProvider<BatchNormLayer> build = GradientLayerProviderBuilder.init()
                .withModel(neuralNetwork)
                .withIntermediateResults(Collections.singletonList(new IntermediateOutputResult(DoubleMatrix.ones(2))))
                .build();
        Assertions.assertTrue(build instanceof GradientBatchNormLayerProvider);
    }

    @Test
    void instanciate_weight_bias_provider() {
        NeuralNetworkModel<WeightBiasLayer> neuralNetwork =
                NeuralNetworkModelBuilder.init()
                        .input(2)
                        .addLayer(2)
                        .buildWeightBiasModel();

        GradientLayerProvider<WeightBiasLayer> build = GradientLayerProviderBuilder.init()
                .withModel(neuralNetwork)
                .withResults(Collections.singletonList(DoubleMatrix.ones(2)))
                .build();
        Assertions.assertTrue(build instanceof GradientWeightBiasLayerProvider);
    }
}