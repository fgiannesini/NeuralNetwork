package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.stream.Collectors;

public class GradientWeightBiasLayerProvider extends GradientLayerProvider {

    private final List<WeightBiasLayer> layers;

    public GradientWeightBiasLayerProvider(List<Layer> layers, List<DoubleMatrix> results) {
        super(results, layers.size());
        this.layers = layers.stream().map(WeightBiasLayer.class::cast).collect(Collectors.toList());
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return layers.get(currentLayerIndex).getWeightMatrix();
    }

    public ActivationFunctionApplier getCurrentActivationFunction() {
        return layers.get(currentLayerIndex - 1).getActivationFunctionType().getActivationFunction();
    }
}
