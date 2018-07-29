package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import org.jblas.DoubleMatrix;

public class GradientDescentCorrection {
    private final DoubleMatrix weightCorrectionResults;
    private final DoubleMatrix biasCorrectionResults;

    public GradientDescentCorrection(DoubleMatrix weightCorrectionResults, DoubleMatrix biasCorrectionResults) {
        this.weightCorrectionResults = weightCorrectionResults;
        this.biasCorrectionResults = biasCorrectionResults;
    }

    public DoubleMatrix getWeightCorrectionResults() {
        return weightCorrectionResults;
    }

    public DoubleMatrix getBiasCorrectionResults() {
        return biasCorrectionResults;
    }
}
