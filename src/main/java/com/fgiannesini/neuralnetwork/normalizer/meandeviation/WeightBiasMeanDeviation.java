package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import org.jblas.DoubleMatrix;

public class WeightBiasMeanDeviation extends MeanDeviation {

    public WeightBiasMeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        super(mean, deviation);
    }
}
