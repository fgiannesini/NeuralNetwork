package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

public interface LearningAlgorithm {

    NeuralNetworkModel learn(LayerTypeData inputData, LayerTypeData outputData);

    void updateLearningRate(double learningRate);
}
