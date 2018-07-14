package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class NeuralNetwork {

    private final LearningAlgorithm learningAlgorithm;
    private NeuralNetworkModel neuralNetworkModel;

    NeuralNetwork(LearningAlgorithm learningAlgorithm) {
        this.learningAlgorithm = learningAlgorithm;
    }

    void learn(double[] input, double[] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromTabToDoubleMatrix(expected);
        learn(inputMatrix, y);
    }

    void learn(double[][] input, double[][] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        learn(inputMatrix, y);
    }

    public void learn(DoubleMatrix input, DoubleMatrix outpout) {
        neuralNetworkModel = learningAlgorithm.learn(input, outpout);
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
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer()
                .compute(input);
    }

}
