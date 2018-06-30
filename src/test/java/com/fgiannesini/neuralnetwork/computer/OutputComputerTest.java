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
    void compute_one_dimension_output_with_one_hidden_layer() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(4, ActivationFunctionType.NONE)
                .output(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        float[] inputData = new float[3];
        Arrays.fill(inputData, 1);

        float[] output = OutputComputerBuilder.init()
                .withModel(model)
                .build()
                .compute(inputData);
        Assertions.assertArrayEquals(new float[]{17, 17}, output);
    }

    @Test
    void compute_one_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .output(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        float[] inputData = new float[3];
        Arrays.fill(inputData, 1);

        float[] output = OutputComputerBuilder.init()
                .withModel(model)
                .build()
                .compute(inputData);
        Assertions.assertArrayEquals(new float[]{39, 39}, output);
    }

    @Test
    void compute_two_dimension_output_with_three_hidden_layers() {
        NeuralNetworkModel model = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .addLayer(2, ActivationFunctionType.NONE)
                .output(2, ActivationFunctionType.NONE)
                .useInitializer(InitializerType.ONES)
                .build();

        float[][] inputData = {
                {1, 1, 1},
                {2, 2, 2}
        };

        float[][] output = OutputComputerBuilder.init()
                .withModel(model)
                .build()
                .compute(inputData);
        float[][] expected = {{39, 39}, {63, 63}};
        Assertions.assertArrayEquals(expected, output);
    }
}