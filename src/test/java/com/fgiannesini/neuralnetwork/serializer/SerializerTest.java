package com.fgiannesini.neuralnetwork.serializer;

import com.fgiannesini.neuralnetwork.HyperParameters;
import com.fgiannesini.neuralnetwork.RegularizationCoeffs;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.learningrate.LearningRateUpdaterType;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

class SerializerTest {

    @Test
    void serialize_neural_network_model() throws IOException {
        NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                .input(28, 28, 1)
                .addConvolutionLayer(5, 0, 1, 3, ActivationFunctionType.RELU)
                .addAveragePoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addFullyConnectedLayer(10, ActivationFunctionType.SOFT_MAX)
                .buildConvolutionNetworkModel();

        File file = File.createTempFile("neural", "network");
        file.deleteOnExit();
        Serializer.get().serialize(neuralNetworkModel, file);

        NeuralNetworkModel deserializedNeuralNetworkModel = Serializer.get().deserialize(file);

        NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, deserializedNeuralNetworkModel);
    }

    @Test
    void serialize_hyper_parameters() throws IOException {
        HyperParameters hyperParameters = new HyperParameters()
                .convolutionLayers(new int[]{1, 2})
                .hiddenLayerSize(new int[]{3, 4})
                .layerType(LayerType.POOLING_MAX)
                .regularizationCoeff(new RegularizationCoeffs())
                .batchSize(10)
                .epochCount(1)
                .learningRateUpdater(LearningRateUpdaterType.CONSTANT.get(0.01))
                .momentumCoeff(0.96D)
                .rmsStopCoeff(0.39D);
        File file = File.createTempFile("hyper", "parameters");
        file.deleteOnExit();
        Serializer.get().serialize(hyperParameters, file);
        HyperParameters deserializedHyperParameters = Serializer.get().deserialize(file);
        Assertions.assertEquals(deserializedHyperParameters, hyperParameters);
    }

}