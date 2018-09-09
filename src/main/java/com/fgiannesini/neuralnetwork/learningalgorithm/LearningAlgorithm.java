package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public interface LearningAlgorithm<L extends Layer> {

    default NeuralNetworkModel<L> learn(double[] input, double[] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromTabToDoubleMatrix(expected);
        return learn(inputMatrix, y);
    }

    default NeuralNetworkModel<L> learn(double[][] input, double[][] expected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix y = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        return learn(inputMatrix, y);
    }

    NeuralNetworkModel<L> learn(DoubleMatrix inputMatrix, DoubleMatrix y);

    void updateLearningRate(double learningRate);
}
