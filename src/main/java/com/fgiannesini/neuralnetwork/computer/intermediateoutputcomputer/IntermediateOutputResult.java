package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviation;
import org.jblas.DoubleMatrix;

public class IntermediateOutputResult {

    private final LayerTypeData result;
    private MeanDeviation meanDeviation;
    private DoubleMatrix beforeNormalisationResult;
    private DoubleMatrix afterMeanApplicationResult;

    public IntermediateOutputResult(LayerTypeData result, MeanDeviation meanDeviation, DoubleMatrix beforeNormalisationResult, DoubleMatrix afterMeanApplicationResult) {
        this.result = result;
        this.meanDeviation = meanDeviation;
        this.beforeNormalisationResult = beforeNormalisationResult;
        this.afterMeanApplicationResult = afterMeanApplicationResult;
    }

    public IntermediateOutputResult(LayerTypeData input) {
        this.result = input;
    }

    public LayerTypeData getResult() {
        return result;
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
