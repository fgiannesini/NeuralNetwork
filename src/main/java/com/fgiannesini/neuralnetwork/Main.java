package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1)
                .addLayer(10, ActivationFunctionType.RELU)
                .addLayer(10, ActivationFunctionType.SIGMOID)
                .build();

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .build();

        int learningSize = 100;
        double[] input = new Random().doubles(learningSize).map(d -> d * 10).toArray();
        DoubleMatrix inputMatrix = new DoubleMatrix(1, learningSize, input);
        double[] output = Arrays.stream(input).map(Math::round).toArray();
        DoubleMatrix outputMatrix = new DoubleMatrix(1, learningSize, output);

        int testSize = 20;
        double[] testInput = new Random().doubles(testSize).map(d -> d * 10).toArray();
        DoubleMatrix testInputMatrix = new DoubleMatrix(1, testSize, testInput);
        double[] testOutput = Arrays.stream(testInput).map(Math::round).toArray();
        DoubleMatrix testOutputMatrix = new DoubleMatrix(1, testSize, testOutput);

        neuralNetwork.learn(inputMatrix, outputMatrix, testInputMatrix, testOutputMatrix);
    }
}
