package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class ConvolutionNeuralNetworkModelBuilder {

    private final List<Integer> filterSizes;
    private final List<Integer> paddings;
    private final List<Integer> strides;
    private final List<Integer> channelCounts;

    private final List<ActivationFunctionType> layerActivationFunctions;
    private final List<LayerType> layerTypes;

    private int inputWidth;
    private int inputHeight;

    private InitializerType initializerType;

    private ConvolutionNeuralNetworkModelBuilder() {
        filterSizes = new ArrayList<>();
        paddings = new ArrayList<>();
        strides = new ArrayList<>();
        channelCounts = new ArrayList<>();
        initializerType = InitializerType.XAVIER;
        layerActivationFunctions = new ArrayList<>();
        layerTypes = new ArrayList<>();
    }

    public static ConvolutionNeuralNetworkModelBuilder init() {
        return new ConvolutionNeuralNetworkModelBuilder();
    }

    public ConvolutionNeuralNetworkModelBuilder input(int inputWidth, int inputHeight) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addMaxPoolingLayer(int filterSize, int padding, int stride, ActivationFunctionType activationFunctionType) {
        filterSizes.add(filterSize);
        paddings.add(padding);
        strides.add(stride);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.POOLING_MAX);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addAveragePoolingLayer(int filterSize, int padding, int stride, ActivationFunctionType activationFunctionType) {
        filterSizes.add(filterSize);
        paddings.add(padding);
        strides.add(stride);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.POOLING_AVERAGE);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addFullyConnectedLayer(int layerNodeCount, ActivationFunctionType activationFunctionType) {
        filterSizes.add(layerNodeCount);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.FULLY_CONNECTED);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addConvolutionLayer(int filterSize, int padding, int stride, int outputChannelCount, ActivationFunctionType activationFunctionType) {
        filterSizes.add(filterSize);
        paddings.add(padding);
        strides.add(stride);
        channelCounts.add(outputChannelCount);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.CONVOLUTION);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder useInitializer(InitializerType type) {
        this.initializerType = type;
        return this;
    }

    private NeuralNetworkModel buildConvolutionNetworkModel() {
        checkInputs();

        int outputSize = layerNodeCounts.get(layerNodeCounts.size() - 1);
        NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(inputWidth, outputSize);
        Initializer initializer = initializerType.getInitializer();
        WeightBiasLayer firstWeightBiasLayer = new WeightBiasLayer(inputWidth, layerNodeCounts.get(0), initializer, layerActivationFunctions.get(0));
        neuralNetworkModel.addLayer(firstWeightBiasLayer);
        IntStream.range(1, layerNodeCounts.size()).forEach(i -> {
            Integer inputLayerSize = layerNodeCounts.get(i - 1);
            Integer outputLayerSize = layerNodeCounts.get(i);
            WeightBiasLayer weightBiasLayer = new WeightBiasLayer(inputLayerSize, outputLayerSize, initializer, layerActivationFunctions.get(i));
            neuralNetworkModel.addLayer(weightBiasLayer);
        });
        return neuralNetworkModel;
    }

    private void checkInputs() {
        if (inputWidth <= 0) {
            throw new IllegalArgumentException("Size of input data should be set");
        }
        if (layerNodeCounts.isEmpty()) {
            throw new IllegalArgumentException("At least one hidden layer should be set");
        }
    }

}
