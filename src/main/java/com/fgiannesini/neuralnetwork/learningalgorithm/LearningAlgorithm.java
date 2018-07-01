package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public interface LearningAlgorithm {

    default NeuralNetworkModel learn(double[] input, double[] expected) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input);
        DoubleMatrix y = new DoubleMatrix(expected);
        return learn(inputMatrix, y);
    }

    default NeuralNetworkModel learn(double[][] input, double[][] expected) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input).transpose();
        DoubleMatrix y = new DoubleMatrix(expected).transpose();
        return learn(inputMatrix, y);
    }

    NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y);
}
