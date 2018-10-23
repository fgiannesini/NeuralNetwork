package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import org.jblas.DoubleMatrix;

public class WeightBiasMeanDeviation implements MeanDeviation {

    private final DoubleMatrix mean;
    private final DoubleMatrix deviation;

    public WeightBiasMeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        this.mean = mean;
        this.deviation = deviation;
    }

    public DoubleMatrix getMean() {
        return mean;
    }

    public DoubleMatrix getDeviation() {
        return deviation;
    }
}
