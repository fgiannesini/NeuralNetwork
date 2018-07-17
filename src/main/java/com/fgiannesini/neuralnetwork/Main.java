package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1)
                .addLayer(10, ActivationFunctionType.RELU)
                .addLayer(10, ActivationFunctionType.RELU)
                .addLayer(10, ActivationFunctionType.SIGMOID)
                .build();

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .build();

        int learningSize = 1000;
        double[] input = new Random().doubles(learningSize).map(d -> d * 10).toArray();
        DoubleMatrix inputMatrix = new DoubleMatrix(1, learningSize, input);
        DoubleMatrix outputMatrix = convertToOutputFormat(input);

        int testSize = 100;
        double[] testInput = new Random().doubles(testSize).map(d -> d * 10).toArray();
        DoubleMatrix testInputMatrix = new DoubleMatrix(1, testSize, testInput);
        DoubleMatrix testOutputMatrix = convertToOutputFormat(testInput);

        neuralNetwork.learn(inputMatrix, outputMatrix, testInputMatrix, testOutputMatrix);

        DoubleMatrix testOutputPredictionMatrix = neuralNetwork.apply(testInputMatrix);
        testOutputPredictionMatrix = convertToOutputFormat(testOutputPredictionMatrix.columnArgmaxs());
        double error = MatrixFunctions.abs(testOutputMatrix.sub(testOutputPredictionMatrix)).sum() / 2d;
        System.out.println("Success Rate = " + (1 - error / (double) testSize) * 100d + " %");
    }

    public static DoubleMatrix convertToOutputFormat(double[] input) {
        int[] outputValues = Arrays.stream(input).map(Math::floor).mapToInt(d -> (int) d).toArray();
        return convertToOutputFormat(outputValues);
    }

    public static DoubleMatrix convertToOutputFormat(int[] outputValues) {
        int learningSize = outputValues.length;
        double[][] output = new double[learningSize][10];
        for (int i = 0; i < learningSize; i++) {
            output[i] = new double[10];
            output[i][outputValues[i]] = 1;
        }
        return DataFormatConverter.fromDoubleTabToDoubleMatrix(output);
    }
}
