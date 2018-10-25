package com.fgiannesini.neuralnetwork.learningrate;

import java.io.Serializable;

public interface ILearningRateUpdater extends Serializable {

    double get(int epochNumber);
}
