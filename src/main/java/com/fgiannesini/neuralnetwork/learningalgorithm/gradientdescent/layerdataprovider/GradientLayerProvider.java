package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

public abstract class GradientLayerProvider {

    private final DoubleMatrix results;
    private final DoubleMatrix previousResults;
    private final ActivationFunctionType activationFunctionType;
    private final int layerIndex;

    GradientLayerProvider(DoubleMatrix results, DoubleMatrix previousResults, ActivationFunctionType activationFunctionType, int layerIndex) {
        this.results = results;
        this.previousResults = previousResults;
        this.activationFunctionType = activationFunctionType;
        this.layerIndex = layerIndex;
    }

    public DoubleMatrix getPreviousResult() {
        return previousResults;
    }

    public DoubleMatrix getCurrentResult() {
        return results;
    }

    public ActivationFunctionApplier getActivationFunction() {
        return activationFunctionType.getActivationFunction();
    }

    public int getLayerIndex() {
        return layerIndex;
    }
}
