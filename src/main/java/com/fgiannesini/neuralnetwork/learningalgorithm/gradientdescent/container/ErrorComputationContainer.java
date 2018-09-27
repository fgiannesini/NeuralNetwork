package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

public class ErrorComputationContainer {
    private final GradientLayerProvider provider;
    private final LayerTypeData previousError;
    private final int currentLayerIndex;

    public ErrorComputationContainer(GradientLayerProvider provider, LayerTypeData previousError, int currentLayerIndex) {
        this.provider = provider;
        this.previousError = previousError;
        this.currentLayerIndex = currentLayerIndex;
    }

    public GradientLayerProvider getProvider() {
        return provider;
    }

    public LayerTypeData getPreviousError() {
        return previousError;
    }

    public int getCurrentLayerIndex() {
        return currentLayerIndex;
    }
}
