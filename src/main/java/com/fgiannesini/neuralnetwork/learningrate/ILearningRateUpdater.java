package com.fgiannesini.neuralnetwork.learningrate;

public interface ILearningRateUpdater {

    double get(int epochNumber);
}
