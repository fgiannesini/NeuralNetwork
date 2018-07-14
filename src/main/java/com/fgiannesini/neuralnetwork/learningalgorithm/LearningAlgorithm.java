package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public interface LearningAlgorithm {

    default NeuralNetworkModel learn(double[] input, double[] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromTabToDoubleMatrix(expected);
        return learn(inputMatrix, y);
    }

    default NeuralNetworkModel learn(double[][] input, double[][] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        return learn(inputMatrix, y);
    }

    NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y);
}
