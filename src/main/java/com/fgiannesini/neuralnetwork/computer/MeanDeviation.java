package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

public class MeanDeviation {
    private DoubleMatrix mean;
    private DoubleMatrix deviation;

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
