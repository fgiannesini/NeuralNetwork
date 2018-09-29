package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class GradientWeightBiasLayerProvider extends GradientLayerProvider {

    private final WeightBiasLayer previousLayer;

    public GradientWeightBiasLayerProvider(WeightBiasLayer layer, WeightBiasLayer previousLayer, DoubleMatrix results, DoubleMatrix previousResult, int layerIndex) {
        super(results, previousResult, layer.getActivationFunctionType(), layerIndex);
        this.previousLayer = previousLayer;
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return previousLayer.getWeightMatrix();
    }

}
