package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;

public interface INormalizer {

    DoubleMatrix normalize(DoubleMatrix input);
}
