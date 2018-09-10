package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.DoubleMatrix;

public interface ActivationFunctionApplier {

    DoubleMatrix apply(DoubleMatrix input);

    DoubleMatrix derivate(DoubleMatrix input, DoubleMatrix previousError);
}
