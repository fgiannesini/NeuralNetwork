package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class GradientWeightBiasLayerProvider extends GradientLayerProvider {

    private final WeightBiasLayer layer;

    public GradientWeightBiasLayerProvider(WeightBiasLayer layer, DoubleMatrix results, DoubleMatrix previousResult) {
        super(results, previousResult, layer.getActivationFunctionType());
        this.layer = layer;
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return layer.getWeightMatrix();
    }

}
