package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetworkModelBuilder {

    private final List<Integer> layerNodeCounts;
    private final List<ActivationFunctionType> layerActivationFunctions;
    private int inputSize;
    private InitializerType initializerType;

    private NeuralNetworkModelBuilder() {
        layerNodeCounts = new ArrayList<>();
        initializerType = InitializerType.RANDOM;
        layerActivationFunctions = new ArrayList<>();
    }

    public static NeuralNetworkModelBuilder init() {
        return new NeuralNetworkModelBuilder();
    }

    public NeuralNetworkModelBuilder input(int inputSize) {
        this.inputSize = inputSize;
        return this;
    }

    public NeuralNetworkModelBuilder addLayer(int layerNodeCount) {
        layerNodeCounts.add(layerNodeCount);
        layerActivationFunctions.add(ActivationFunctionType.RELU);
        return this;
    }

    public NeuralNetworkModelBuilder addLayer(int layerNodeCount, ActivationFunctionType activationFunctionType) {
        layerNodeCounts.add(layerNodeCount);
        layerActivationFunctions.add(activationFunctionType);
        return this;
    }

    public NeuralNetworkModelBuilder useInitializer(InitializerType type) {
        this.initializerType = type;
        return this;
    }

    public NeuralNetworkModel build() {
        checkInputs();
        return buildNeuralNetworkModel();
    }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        int outputSize = layerNodeCounts.get(layerNodeCounts.size() - 1);
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(inputSize, outputSize, initializerType.getInitializer());

        neuralNetworkModel.addLayer(inputSize, layerNodeCounts.get(0), layerActivationFunctions.get(0));
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            neuralNetworkModel.addLayer(inputLayerSize, outputLayerSize, layerActivationFunctions.get(i));
        });
        return neuralNetworkModel;
    }

    private void checkInputs() {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("Size of input data should be set");
        }
        if (layerNodeCounts.isEmpty()) {
            throw new IllegalArgumentException("At least one hidden layer should be set");
        }
    }
}
