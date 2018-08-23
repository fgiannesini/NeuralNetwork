package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientLayerProvider {

    private final List<DoubleMatrix> results;
    private final List<WeightBiasLayer> layers;
    private int currentLayerIndex;

    public GradientLayerProvider(List<WeightBiasLayer> layers, List<DoubleMatrix> results) {
        currentLayerIndex = layers.size();
        this.results = results;
        this.layers = layers;
    }

    public void nextLayer() {
        currentLayerIndex--;
    }

    public boolean hasNextLayer() {
        return currentLayerIndex > 0;
    }

    public DoubleMatrix getPreviousWeightMatrix() {
        return layers.get(currentLayerIndex).getWeightMatrix();
    }

    public ActivationFunctionApplier getCurrentActivationFunction() {
        return layers.get(currentLayerIndex - 1).getActivationFunctionType().getActivationFunction();
    }

    public DoubleMatrix getPreviousResult() {
        return results.get(currentLayerIndex - 1);
    }

    public DoubleMatrix getCurrentResult() {
        return results.get(currentLayerIndex);
    }

    public int getCurrentLayerIndex() {
        return currentLayerIndex;
    }

}
