package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.Layer;

public class GradientConvolutionLayerProvider extends GradientLayerProvider {
    public GradientConvolutionLayerProvider(Layer layer, LayerTypeData results, LayerTypeData previousResult, int layerIndex) {
        super(results, previousResult, layer, layerIndex, layer.getActivationFunctionType());
    }
}
