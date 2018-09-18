package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class NeuralNetworkModelBuilder {

    private final List<Integer> layerNodeCounts;
    private final List<ActivationFunctionType> layerActivationFunctions;
    private final List<LayerType> layerTypes;

    private int inputSize;
    private InitializerType initializerType;

    private NeuralNetworkModelBuilder() {
        layerNodeCounts = new ArrayList<>();
        initializerType = InitializerType.XAVIER;
        layerActivationFunctions = new ArrayList<>();
        layerTypes = new ArrayList<>();
    }

    public static NeuralNetworkModelBuilder init() {
        return new NeuralNetworkModelBuilder();
    }

    public NeuralNetworkModelBuilder input(int inputSize) {
        this.inputSize = inputSize;
        return this;
    }

    public NeuralNetworkModelBuilder addWeightBiasLayer(int layerNodeCount, ActivationFunctionType activationFunctionType) {
        layerNodeCounts.add(layerNodeCount);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.WEIGHT_BIAS);
        return this;
    }

    public NeuralNetworkModelBuilder addBatchNormLayer(int layerNodeCount, ActivationFunctionType activationFunctionType) {
        layerNodeCounts.add(layerNodeCount);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.BATCH_NORM);
        return this;
    }

    public NeuralNetworkModelBuilder useInitializer(InitializerType type) {
        this.initializerType = type;
        return this;
    }

    public NeuralNetworkModel buildWeightBiasModel() {
        checkInputs();
        return buildNeuralNetworkModel();
    }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        checkInputs();

        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel();
        Initializer initializer = initializerType.getInitializer();
        Layer firstLayer = buildLayerInstance(initializer, inputSize, layerNodeCounts.get(0), layerActivationFunctions.get(0), layerTypes.get(0));
        neuralNetworkModel.addLayer(firstLayer);
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            Layer layer = buildLayerInstance(initializer, inputLayerSize, outputLayerSize, layerActivationFunctions.get(i), layerTypes.get(i));
            neuralNetworkModel.addLayer(layer);
        });
        return neuralNetworkModel;
    }

    private Layer buildLayerInstance(Initializer initializer, int inputLayerSize, Integer outputLayerSize, ActivationFunctionType activationFunctionType, LayerType layerType) {
        Layer layer;
        switch (layerType) {
            case WEIGHT_BIAS:
                layer = new WeightBiasLayer(inputLayerSize, outputLayerSize, initializer, activationFunctionType);
                break;
            case BATCH_NORM:
                layer = new BatchNormLayer(inputLayerSize, outputLayerSize, initializer, activationFunctionType);
                break;
            default:
                throw new IllegalArgumentException(layerType + " is not allowed for neural network model");
        }
        return layer;
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
