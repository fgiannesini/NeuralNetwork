package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import org.jblas.DoubleMatrix;

public abstract class MeanDeviation {
    private final DoubleMatrix mean;
    private final DoubleMatrix deviation;

    public MeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        this.mean = mean;
        this.deviation = deviation;
    }

    public DoubleMatrix getMean() {
        return mean;
    }

    public DoubleMatrix getDeviation() {
        return deviation;
    }

    @Override
    public String toString() {
        return "MeanDeviation{" +
                "mean=" + mean +
                ", deviation=" + deviation +
                '}';
    }
}
