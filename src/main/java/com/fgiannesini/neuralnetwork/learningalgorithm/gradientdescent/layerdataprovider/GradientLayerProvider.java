package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

public abstract class GradientLayerProvider {

    private final DoubleMatrix results;
    private final DoubleMatrix previousResults;

    GradientLayerProvider(DoubleMatrix results, DoubleMatrix previousResults, ActivationFunctionType activationFunctionType) {
        this.results = results;
        this.previousResults = previousResults;
    }

    public DoubleMatrix getPreviousResult() {
        return previousResults;
    }

    public DoubleMatrix getCurrentResult() {
        return results;
    }
}
