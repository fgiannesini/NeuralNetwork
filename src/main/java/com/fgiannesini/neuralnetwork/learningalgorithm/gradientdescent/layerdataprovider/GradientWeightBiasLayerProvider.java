package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;

public class GradientWeightBiasLayerProvider extends GradientLayerProvider {

    public GradientWeightBiasLayerProvider(WeightBiasLayer layer, WeightBiasData results, WeightBiasData previousResult, int layerIndex) {
        super(results, previousResult, layer, layerIndex, layer.getActivationFunctionType());
    }

}
