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

    public NeuralNetworkModel<WeightBiasLayer> buildWeightBiasModel() {
        checkInputs();
        return buildWeightBiasNeuralNetworkModel();
    }

    private NeuralNetworkModel<WeightBiasLayer> buildWeightBiasNeuralNetworkModel() {
        int outputSize = layerNodeCounts.get(layerNodeCounts.size() - 1);
        NeuralNetworkModel<WeightBiasLayer> neuralNetworkModel = new NeuralNetworkModel<>(inputSize, outputSize, LayerType.WEIGHT_BIAS);
        Initializer initializer = initializerType.getInitializer();
        WeightBiasLayer firstWeightBiasLayer = new WeightBiasLayer(inputSize, layerNodeCounts.get(0), initializer, layerActivationFunctions.get(0));
        neuralNetworkModel.addLayer(firstWeightBiasLayer);
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            WeightBiasLayer weightBiasLayer = new WeightBiasLayer(inputLayerSize, outputLayerSize, initializer, layerActivationFunctions.get(i));
            neuralNetworkModel.addLayer(weightBiasLayer);
        });
        return neuralNetworkModel;
    }

    public NeuralNetworkModel<BatchNormLayer> buildBatchNormModel() {
        checkInputs();
        return buildBatchNormNeuralNetworkModel();
    }

    private NeuralNetworkModel<BatchNormLayer> buildBatchNormNeuralNetworkModel() {
        int outputSize = layerNodeCounts.get(layerNodeCounts.size() - 1);
        NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = new NeuralNetworkModel<>(inputSize, outputSize, LayerType.BATCH_NORM);
        Initializer initializer = initializerType.getInitializer();
        BatchNormLayer firstBatchNormLayer = new BatchNormLayer(inputSize, layerNodeCounts.get(0), initializer, layerActivationFunctions.get(0));
        neuralNetworkModel.addLayer(firstBatchNormLayer);
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            BatchNormLayer batchNormLayer = new BatchNormLayer(inputLayerSize, outputLayerSize, initializer, layerActivationFunctions.get(i));
            neuralNetworkModel.addLayer(batchNormLayer);
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
