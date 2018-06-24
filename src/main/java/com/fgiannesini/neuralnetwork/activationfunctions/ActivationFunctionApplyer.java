package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public interface ActivationFunctionApplyer {

    FloatMatrix apply(FloatMatrix input);
}
