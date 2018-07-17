package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.INormalizer;
import org.jblas.DoubleMatrix;

public class NeuralNetwork {

    private final LearningAlgorithm learningAlgorithm;
    private final INormalizer normalizer;
    private final CostType costType;
    private NeuralNetworkModel neuralNetworkModel;
    private int learningIterationCount;

    NeuralNetwork(LearningAlgorithm learningAlgorithm, INormalizer normalizer, CostType costType) {
        this.learningAlgorithm = learningAlgorithm;
        this.normalizer = normalizer;
        this.costType = costType;
        this.learningIterationCount = 100;
    }

    void learn(double[] input, double[] expected, double[] testInput, double[] testExpected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromTabToDoubleMatrix(expected);
        DoubleMatrix testInputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix testExpectedMatrix = DataFormatConverter.fromTabToDoubleMatrix(expected);
        learn(inputMatrix, outputMatrix, testInputMatrix, testExpectedMatrix);
    }

    void learn(double[][] input, double[][] expected, double[][] testInput, double[][] testExpected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        DoubleMatrix testInputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix testExpectedMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        learn(inputMatrix, outputMatrix, testInputMatrix, testExpectedMatrix);
    }

    public void learn(DoubleMatrix input, DoubleMatrix outpout, DoubleMatrix testInput, DoubleMatrix testOutpout) {
        DoubleMatrix normalizedInput = normalizer.normalize(input);
        DoubleMatrix normalizedOutput = normalizer.normalize(outpout);
        DoubleMatrix normalizedTestInput = normalizer.normalize(testInput);
        DoubleMatrix normalizedTestOutput = normalizer.normalize(testOutpout);
        for (int i = 0; i < learningIterationCount; i++) {
            neuralNetworkModel = learningAlgorithm.learn(normalizedInput, normalizedOutput);
            CostComputer costComputer = CostComputerBuilder.init()
                    .withNeuralNetworkModel(neuralNetworkModel)
                    .withType(costType)
                    .build();
            double inputCost = costComputer.compute(normalizedInput, normalizedTestInput);
            double outputCost = costComputer.compute(normalizedInput, normalizedTestOutput);
            System.out.println("inputCost = " + inputCost);
            System.out.println("outputCost = " + outputCost);
            System.out.println();
        }
    }

    public double[] apply(double[] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input);
        return apply(inputMatrix).toArray();
    }

    public double[][] apply(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input);
        return apply(inputMatrix).transpose().toArray2();
    }

    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix normalizedInput = normalizer.normalize(input);
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer()
                .compute(normalizedInput);
    }

}
