package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

public interface LearningAlgorithm {

    default NeuralNetworkModel learn(float[] input, float[] expected) {
        FloatMatrix inputMatrix = new FloatMatrix(input);
        FloatMatrix y = new FloatMatrix(expected);
        return learn(inputMatrix, y);
    }

    default NeuralNetworkModel learn(float[][] input, float[][] expected) {
        FloatMatrix inputMatrix = new FloatMatrix(input).transpose();
        FloatMatrix y = new FloatMatrix(expected).transpose();
        return learn(inputMatrix, y);
    }

    NeuralNetworkModel learn(FloatMatrix inputMatrix, FloatMatrix y);
}
