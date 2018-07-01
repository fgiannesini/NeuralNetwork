package com.fgiannesini.neuralnetwork.learningalgorithm;

import org.jblas.DoubleMatrix;

class GradientDescentCorrection {
    private DoubleMatrix weightCorrectionResults;
    private DoubleMatrix biasCorrectionResults;

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
