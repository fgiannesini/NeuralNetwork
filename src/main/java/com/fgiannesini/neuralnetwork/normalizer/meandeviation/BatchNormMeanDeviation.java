package com.fgiannesini.neuralnetwork.normalizer.meandeviation;

import org.jblas.DoubleMatrix;

public class BatchNormMeanDeviation extends MeanDeviation {

    public BatchNormMeanDeviation(DoubleMatrix mean, DoubleMatrix deviation) {
        super(mean, deviation);
    }
}
