package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider;

import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientWeightBiasLayerProvider extends GradientLayerProvider<WeightBiasLayer> {

    public GradientWeightBiasLayerProvider(List<WeightBiasLayer> layers, List<DoubleMatrix> results) {
        super(layers, results);
    }
}
