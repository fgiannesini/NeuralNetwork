package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkModel {

  private final int inputSize;
  private final int outputSize;
  private Initializer initializer;
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
}
