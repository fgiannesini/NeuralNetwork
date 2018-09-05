package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import org.jblas.DoubleMatrix;

public class IntermediateOutputResult {

    private DoubleMatrix result;
    private MeanDeviation meanDeviation;
    private DoubleMatrix beforeNormalisationResult;
    private DoubleMatrix afterMeanApplicationResult;

    public IntermediateOutputResult(DoubleMatrix result, MeanDeviation meanDeviation, DoubleMatrix beforeNormalisationResult, DoubleMatrix afterMeanApplicationResult) {
        this.result = result;
        this.meanDeviation = meanDeviation;
        this.beforeNormalisationResult = beforeNormalisationResult;
        this.afterMeanApplicationResult = afterMeanApplicationResult;
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

    public DoubleMatrix getBeforeNormalisationResult() {
        return beforeNormalisationResult;
    }

    public DoubleMatrix getAfterMeanApplicationResult() {
        return afterMeanApplicationResult;
    }
}
