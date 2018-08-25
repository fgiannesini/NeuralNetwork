package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class NeuralNetworkModelTest {

    @Test
    void clone_WeightBiasNeuralNetworkModel() {
        Initializer initializer = InitializerType.ZEROS.getInitializer();
        NeuralNetworkModel<WeightBiasLayer> neuralNetworkModel = new NeuralNetworkModel<>(3, 2, LayerType.WEIGHT_BIAS);
        WeightBiasLayer weightBiasLayer = new WeightBiasLayer(3, 2, initializer, ActivationFunctionType.NONE);
        neuralNetworkModel.addLayer(weightBiasLayer);

        NeuralNetworkModel clone = neuralNetworkModel.clone();

        assertClone(neuralNetworkModel, clone);
    }

    @Test
    void clone_BatchNormNeuralNetworkModel() {
        Initializer initializer = InitializerType.ZEROS.getInitializer();
        NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = new NeuralNetworkModel<>(3, 2, LayerType.WEIGHT_BIAS);
        BatchNormLayer batchNormLayer = new BatchNormLayer(3, 2, initializer, ActivationFunctionType.NONE);
        neuralNetworkModel.addLayer(batchNormLayer);

        NeuralNetworkModel clone = neuralNetworkModel.clone();

        assertClone(neuralNetworkModel, clone);
    }

    private void assertClone(NeuralNetworkModel neuralNetworkModel, NeuralNetworkModel clone) {
        Assertions.assertNotSame(clone, neuralNetworkModel);

        Assertions.assertNotSame(clone.getLayers(), neuralNetworkModel.getLayers());
        Assertions.assertEquals(clone.getLayers(), neuralNetworkModel.getLayers());

        Assertions.assertEquals(clone.getInputSize(), neuralNetworkModel.getInputSize());
        Assertions.assertEquals(clone.getOutputSize(), neuralNetworkModel.getOutputSize());
    }
}