package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;

public class BatchNormMeanDeviation extends MeanDeviation {

    public BatchNormMeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        super(mean, deviation);
    }
}
