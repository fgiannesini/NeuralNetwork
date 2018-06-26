package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class NeuralNetworkModelTest {

    @Test
    void test_clone() {
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(3, 2, InitializerType.ZEROS.getInitializer());
        neuralNetworkModel.addLayer(3, 2, ActivationFunctionType.NONE);

        NeuralNetworkModel clone = neuralNetworkModel.clone();

        Assertions.assertNotSame(clone, neuralNetworkModel);

        Assertions.assertNotSame(clone.getLayers(), neuralNetworkModel.getLayers());
        Assertions.assertEquals(clone.getLayers(), neuralNetworkModel.getLayers());

        Assertions.assertEquals(clone.getInputSize(), neuralNetworkModel.getInputSize());
        Assertions.assertEquals(clone.getOutputSize(), neuralNetworkModel.getOutputSize());
    }
}