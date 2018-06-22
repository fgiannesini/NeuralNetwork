package com.fgiannesini.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkConfigBuilder {

  private int inputSize;
  private int outputSize;
  private List<Integer> layerNodeCounts;

  public static NeuralNetworkConfigBuilder init() {
    return new NeuralNetworkConfigBuilder();
  }

  private NeuralNetworkConfigBuilder() {
    layerNodeCounts = new ArrayList<>();
  }

  public NeuralNetworkConfigBuilder inputSize(int inputSize) {
    this.inputSize = inputSize;
    return this;
  }

  public NeuralNetworkConfigBuilder outputSize(int outputSize) {
    this.outputSize = outputSize;
    return this;
  }

  public NeuralNetworkConfigBuilder addLayer(int layerNodeCount) {
    layerNodeCounts.add(layerNodeCount);
    return this;
  }

  public NeuralNetworkModel build() {
    if (inputSize <= 0) {
      throw new IllegalArgumentException("Size of input data should be set");
    }
    if (outputSize <= 0) {
      throw new IllegalArgumentException("size of output data should be set");
    }
    if (layerNodeCounts.isEmpty()) {
      throw new IllegalArgumentException("At least one hidden layer should be set");
    }
    return new NeuralNetworkModel();
  }
}
