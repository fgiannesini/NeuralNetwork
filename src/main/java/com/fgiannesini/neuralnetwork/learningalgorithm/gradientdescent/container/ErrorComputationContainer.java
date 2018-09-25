package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;

public class ErrorComputationContainer {
    private final GradientLayerProvider provider;
    private final LayerTypeData previousError;

    public ErrorComputationContainer(GradientLayerProvider provider, LayerTypeData previousError) {
        this.provider = provider;
        this.previousError = previousError;
    }

    public GradientLayerProvider getProvider() {
        return provider;
    }

    public LayerTypeData getPreviousError() {
        return previousError;
    }
}
