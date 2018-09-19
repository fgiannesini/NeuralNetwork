package com.fgiannesini.neuralnetwork.model;

import java.util.List;
import java.util.stream.Collectors;

public class NeuralNetworkModel<L extends Layer> implements Cloneable {

    private List<Layer> layers;

    public NeuralNetworkModel(List<Layer> layers) {
        this.layers = layers;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    @Override
    public NeuralNetworkModel clone() {
        try {
            NeuralNetworkModel clone = (NeuralNetworkModel) super.clone();
            clone.layers = this.layers.stream().map(Layer::clone).collect(Collectors.toList());
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return layers.stream().map(Object::toString).collect(Collectors.joining("\n"));
    }
}
