package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

class OutputComputerTest {

    @Test
    void compute_output_One_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .inputSize(3)
                .addLayer(4)
                .outputSize(2)
                .useInitializer(InitializerType.ONES)
                .build();

        float[] inputData = new float[3];
        Arrays.fill(inputData, 1);

        float[] output = OutputComputerBuilder.init()
                .withModel(model)
                .withActivationFunction(ActivationFunctionType.NONE)
                .build()
                .compute(inputData);
        Assertions.assertArrayEquals(new float[]{17, 17}, output);
    }

    @Test
    void compute_output_Three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .inputSize(3)
                .addLayer(2)
                .addLayer(2)
                .addLayer(2)
                .outputSize(2)
                .useInitializer(InitializerType.ONES)
                .build();

        float[] inputData = new float[3];
        Arrays.fill(inputData, 1);

        float[] output = OutputComputerBuilder.init()
                .withModel(model)
                .withActivationFunction(ActivationFunctionType.NONE)
                .build()
                .compute(inputData);
        Assertions.assertArrayEquals(new float[]{39, 39}, output);
    }
}