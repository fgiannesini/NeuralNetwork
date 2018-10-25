package com.fgiannesini.neuralnetwork.serializer;

import com.fgiannesini.neuralnetwork.HyperParameters;
import com.fgiannesini.neuralnetwork.RegularizationCoeffs;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.NeuralNetworkAssertions;
import com.fgiannesini.neuralnetwork.learningrate.LearningRateUpdaterType;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Arrays;

class SerializerTest {

    @AfterEach
    void tearDown() {
        File file = new File("");
        File[] files = file.getParentFile().listFiles(f -> f.getName().endsWith(Serializer.SERIALIZATION_EXTENSION));
        Arrays.stream(files)
                .filter(File::exists)
                .forEach(File::delete);

    }


    @Test
    void serialize_neural_network_model() {
        NeuralNetworkModel neuralNetworkModel = ConvolutionNeuralNetworkModelBuilder.init()
                .input(28, 28, 1)
                .addConvolutionLayer(5, 0, 1, 3, ActivationFunctionType.RELU)
                .addAveragePoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addFullyConnectedLayer(10, ActivationFunctionType.SOFT_MAX)
                .buildConvolutionNetworkModel();

        String fileName = "neuralNetwork";
        Serializer.get().serialize(neuralNetworkModel, fileName);

        NeuralNetworkModel deserializedNeuralNetworkModel = Serializer.get().deserialize(fileName);

        NeuralNetworkAssertions.checkSameNeuralNetworks(neuralNetworkModel, deserializedNeuralNetworkModel);
    }

    @Test
    void serialize_hyper_parameters() {
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
        String fileName = "hyperParameters";
        Serializer.get().serialize(hyperParameters, fileName);
        HyperParameters deserializedHyperParameters = Serializer.get().deserialize(fileName);
        Assertions.assertSame(deserializedHyperParameters, hyperParameters);

    }

}