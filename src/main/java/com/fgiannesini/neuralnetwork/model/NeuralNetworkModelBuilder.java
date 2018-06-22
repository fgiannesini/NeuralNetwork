package com.fgiannesini.neuralnetwork.model;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkModelBuilder {

  private int inputSize;
  private int outputSize;
  private List<Integer> layerNodeCounts;

  public static NeuralNetworkModelBuilder init() {
    return new NeuralNetworkModelBuilder();
  }

  private NeuralNetworkModelBuilder() {
    layerNodeCounts = new ArrayList<>();
  }

  public NeuralNetworkModelBuilder inputSize(int inputSize) {
    this.inputSize = inputSize;
    return this;
  }

  public NeuralNetworkModelBuilder outputSize(int outputSize) {
    this.outputSize = outputSize;
    return this;
  }

  public NeuralNetworkModelBuilder addLayer(int layerNodeCount) {
    layerNodeCounts.add(layerNodeCount);
    return this;
  }

  public NeuralNetworkModel build() {
    checkInputs();
    NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel();

    return neuralNetworkModel;
  }

  private void checkInputs() {
    if (inputSize <= 0) {
      throw new IllegalArgumentException("Size of input data should be set");
    }
    if (outputSize <= 0) {
      throw new IllegalArgumentException("size of output data should be set");
    }
    if (layerNodeCounts.isEmpty()) {
      throw new IllegalArgumentException("At least one hidden layer should be set");
    }
  }
}
