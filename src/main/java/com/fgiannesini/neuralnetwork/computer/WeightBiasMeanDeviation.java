package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

public class WeightBiasMeanDeviation extends MeanDeviation {

    public WeightBiasMeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        super(mean, deviation);
    }
}
