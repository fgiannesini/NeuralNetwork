package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import org.jblas.DoubleMatrix;

public class IntermediateOutputResult {

    private DoubleMatrix result;

    private MeanDeviation meanDeviation;

    public IntermediateOutputResult(DoubleMatrix result, MeanDeviation meanDeviation) {
        this.result = result;
        this.meanDeviation = meanDeviation;
    }

    public IntermediateOutputResult(DoubleMatrix input) {
        this.result = input;
    }

    public DoubleMatrix getResult() {
        return result;
    }

    public void setResult(DoubleMatrix result) {
        this.result = result;
    }

    public MeanDeviation getMeanDeviation() {
        return meanDeviation;
    }
}
