package com.fgiannesini.neuralnetwork.model;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class NeuralNetworkModel<L extends Layer> implements Cloneable {

    private List<L> layers;

    public NeuralNetworkModel() {
        this.layers = new ArrayList<>();
    }

    public void addLayer(L layer) {
        layers.add(layer);
    }

    public List<L> getLayers() {
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
