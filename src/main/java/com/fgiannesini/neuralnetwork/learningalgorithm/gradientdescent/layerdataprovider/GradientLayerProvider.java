package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.Layer;

public abstract class GradientLayerProvider {

    private final LayerTypeData results;
    private final LayerTypeData previousResults;
    private final Layer layer;
    private final ActivationFunctionType activationFunctionType;
    private final int layerIndex;

    GradientLayerProvider(LayerTypeData results, LayerTypeData previousResults, Layer layer, int layerIndex, ActivationFunctionType activationFunctionType) {
        this.results = results;
        this.previousResults = previousResults;
        this.layer = layer;
        this.activationFunctionType = activationFunctionType;
        this.layerIndex = layerIndex;
    }

    public LayerTypeData getPreviousResult() {
        return previousResults;
    }

    public LayerTypeData getCurrentResult() {
        return results;
    }

    public ActivationFunctionApplier getActivationFunction() {
        return activationFunctionType.getActivationFunction();
    }

    public int getLayerIndex() {
        return layerIndex;
    }

    public Layer getLayer() {
        return layer;
    }
}
