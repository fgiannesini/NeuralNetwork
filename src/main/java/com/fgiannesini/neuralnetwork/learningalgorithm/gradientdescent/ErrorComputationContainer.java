package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import org.jblas.DoubleMatrix;

public class ErrorComputationContainer {
    private final GradientLayerProvider provider;
    private final DoubleMatrix previousError;

    public ErrorComputationContainer(GradientLayerProvider provider, DoubleMatrix previousError) {
        this.provider = provider;
        this.previousError = previousError;
    }

    public GradientLayerProvider getProvider() {
        return provider;
    }

    public DoubleMatrix getPreviousError() {
        return previousError;
    }
}