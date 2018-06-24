package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.initializer.InitializerType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetworkModelBuilder {

  private int inputSize;
  private int outputSize;
  private List<Integer> layerNodeCounts;
    private InitializerType initializerType;

  public static NeuralNetworkModelBuilder init() {
    return new NeuralNetworkModelBuilder();
  }

  private NeuralNetworkModelBuilder() {
    layerNodeCounts = new ArrayList<>();
      initializerType = InitializerType.RANDOM;
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

    public NeuralNetworkModelBuilder useInitializer(InitializerType type) {
        this.initializerType = type;
        return this;
    }

  public NeuralNetworkModel build() {
    checkInputs();
      NeuralNetworkModel neuralNetworkModel = buildNeuralNetworkModel();
      return neuralNetworkModel;
  }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(inputSize, outputSize, initializerType.getInitializer());
        neuralNetworkModel.addLayer(inputSize, layerNodeCounts.get(0));
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            neuralNetworkModel.addLayer(inputLayerSize, outputLayerSize);
        });
        neuralNetworkModel.addLayer(layerNodeCounts.get(layerNodeCounts.size() - 1), outputSize);
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
