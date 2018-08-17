package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class NeuralNetworkModel implements Cloneable {

    private final int inputSize;
    private final int outputSize;
    private final Initializer initializer;
    private List<Layer> layers;

    public NeuralNetworkModel(int inputSize, int outputSize, Initializer initializer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.initializer = initializer;
        this.layers = new ArrayList<>();
    }

    public void addLayer(int inputLayerSize, int outputLayerSize, ActivationFunctionType activationFunctionType) {
        layers.add(new Layer(inputLayerSize, outputLayerSize, initializer, activationFunctionType));
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
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
        return layers.stream().map(l -> l.getWeightMatrix().toString() + "\n" + l.getBiasMatrix()).collect(Collectors.joining("\n"));
    }
}
