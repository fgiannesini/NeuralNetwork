package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class GradientDescentCorrection {

    private final List<DoubleMatrix> correctionResults;

    public GradientDescentCorrection(DoubleMatrix... corrections) {
        this.correctionResults = Arrays.stream(corrections).collect(Collectors.toList());
    }

    public void addCorrectionResult(DoubleMatrix correctionResult) {
        correctionResults.add(correctionResult);
    }

    public List<DoubleMatrix> getCorrectionResults() {
        return this.correctionResults;
    }
}
