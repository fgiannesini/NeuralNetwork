package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import org.jblas.DoubleMatrix;

import java.util.List;

public class ConvolutionMeanDeviation implements MeanDeviation {

    private final List<DoubleMatrix> mean;
    private final List<DoubleMatrix> deviation;

    public ConvolutionMeanDeviation(List<DoubleMatrix> mean, List<DoubleMatrix> deviation) {
        this.mean = mean;
        this.deviation = deviation;
    }

    public List<DoubleMatrix> getMean() {
        return mean;
    }

    public List<DoubleMatrix> getDeviation() {
        return deviation;
    }
}
