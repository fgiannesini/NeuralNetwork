package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

class NeuralNetworkModelTest {

    @Test
    void clone_WeightBiasNeuralNetworkModel() {
        Initializer initializer = InitializerType.ZEROS.getInitializer();
        WeightBiasLayer weightBiasLayer = new WeightBiasLayer(3, 2, initializer, ActivationFunctionType.NONE);
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(Collections.singletonList(weightBiasLayer));

        NeuralNetworkModel clone = neuralNetworkModel.clone();

        assertClone(neuralNetworkModel, clone);
    }

    @Test
    void clone_BatchNormNeuralNetworkModel() {
        Initializer initializer = InitializerType.ZEROS.getInitializer();
        BatchNormLayer batchNormLayer = new BatchNormLayer(3, 2, initializer, ActivationFunctionType.NONE);
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(Collections.singletonList(batchNormLayer));

        NeuralNetworkModel clone = neuralNetworkModel.clone();

        assertClone(neuralNetworkModel, clone);
    }

    private void assertClone(NeuralNetworkModel neuralNetworkModel, NeuralNetworkModel clone) {
        Assertions.assertNotSame(clone, neuralNetworkModel);

        Assertions.assertNotSame(clone.getLayers(), neuralNetworkModel.getLayers());
        Assertions.assertEquals(clone.getLayers(), neuralNetworkModel.getLayers());
    }
}