package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class GradientLayerProvider<L extends Layer> {

    private final List<DoubleMatrix> results;
    protected final List<L> layers;
    int currentLayerIndex;

    public GradientLayerProvider(List<L> layers) {
        currentLayerIndex = layers.size();
        this.results = new ArrayList<>();
        this.layers = layers;
    }

    public void addResults(DoubleMatrix result) {
        results.add(result);
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
