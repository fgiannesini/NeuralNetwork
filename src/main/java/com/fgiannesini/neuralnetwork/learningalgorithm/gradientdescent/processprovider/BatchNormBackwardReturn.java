package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import org.jblas.DoubleMatrix;

public class BatchNormBackwardReturn {

    private final DoubleMatrix weightCorrection;
    private final DoubleMatrix dGamma;
    private final DoubleMatrix dBeta;
    private final DoubleMatrix dx;

    public BatchNormBackwardReturn(DoubleMatrix weightCorrection, DoubleMatrix dGamma, DoubleMatrix dBeta, DoubleMatrix dx) {
        this.weightCorrection = weightCorrection;
        this.dGamma = dGamma;
        this.dBeta = dBeta;
        this.dx = dx;
    }

    public GradientDescentCorrection getCorrections() {
        return new GradientDescentCorrection(weightCorrection, dGamma, dBeta);
    }

    public DoubleMatrix getNextError() {
        return dx;
    }
}
