package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.OutputComputer;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

class GradientPropagationLearnerTest {

    @Disabled
    @Test
    void learn_on_vector_with_one_hidden_layer() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.ONES)
                .inputSize(3)
                .addLayer(4)
                .outputSize(2)
                .outputActivationFunction(ActivationFunctionType.NONE)
                .build();
        OutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .build();
        GradientPropagationLearner gradientPropagationLearner = new GradientPropagationLearner(neuralNetworkModel);
        float[] input = new float[]{1f, 2f, 3f};
        float[] output = new float[]{-1f, -2f};

        NeuralNetworkModel optimizedNeuralNetworkModel = gradientPropagationLearner.learn(input, output);

        Assertions.fail("Assertions to be computed");
    }
}