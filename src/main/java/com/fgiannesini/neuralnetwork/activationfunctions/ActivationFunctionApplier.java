package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;

public interface ActivationFunctionApplier {

    FloatMatrix apply(FloatMatrix input);

    FloatMatrix derivate(FloatMatrix input);
}
