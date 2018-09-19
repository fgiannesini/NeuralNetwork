package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
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
    private int inputChannelCount;

    private InitializerType initializerType;
    private NeuralNetworkModelBuilder neuralNetworkModelBuilder;

    private ConvolutionNeuralNetworkModelBuilder() {
        filterSizes = new ArrayList<>();
        paddings = new ArrayList<>();
        strides = new ArrayList<>();
        channelCounts = new ArrayList<>();
        initializerType = InitializerType.XAVIER;
        layerActivationFunctions = new ArrayList<>();
        layerTypes = new ArrayList<>();
        neuralNetworkModelBuilder = NeuralNetworkModelBuilder.init();
    }

    public static ConvolutionNeuralNetworkModelBuilder init() {
        return new ConvolutionNeuralNetworkModelBuilder();
    }

    public ConvolutionNeuralNetworkModelBuilder input(int inputWidth, int inputHeight, int inputChannelCount) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputChannelCount = inputChannelCount;
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addMaxPoolingLayer(int filterSize, int padding, int stride, ActivationFunctionType activationFunctionType) {
        filterSizes.add(filterSize);
        paddings.add(padding);
        strides.add(stride);
        channelCounts.add(0);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.POOLING_MAX);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addAveragePoolingLayer(int filterSize, int padding, int stride, ActivationFunctionType activationFunctionType) {
        filterSizes.add(filterSize);
        paddings.add(padding);
        strides.add(stride);
        channelCounts.add(0);
        layerActivationFunctions.add(activationFunctionType);
        layerTypes.add(LayerType.POOLING_AVERAGE);
        return this;
    }

    public ConvolutionNeuralNetworkModelBuilder addFullyConnectedLayer(int layerNodeCount, ActivationFunctionType activationFunctionType) {
        neuralNetworkModelBuilder
                .addWeightBiasLayer(layerNodeCount, activationFunctionType);
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
        List<Layer> layers = new ArrayList<>();
        Initializer initializer = initializerType.getInitializer();
        Optional<Layer> firstLayer = buildLayerInstance(layerTypes.get(0), initializer, channelCounts.get(0), paddings.get(0), strides.get(0), filterSizes.get(0), layerActivationFunctions.get(0));
        firstLayer.ifPresent(layers::add);
        IntStream.range(1, layerTypes.size()).forEach(i -> {
            Optional<Layer> layer = buildLayerInstance(layerTypes.get(i), initializer, channelCounts.get(i), paddings.get(i), strides.get(i), filterSizes.get(i), layerActivationFunctions.get(i));
            layer.ifPresent(layers::add);
        });

        layers.addAll(neuralNetworkModelBuilder.input(inputWidth * inputHeight * inputChannelCount).buildLayers());

        return new NeuralNetworkModel(layers);
    }

    private Optional<Layer> buildLayerInstance(LayerType layerType, Initializer initializer, Integer channelCount, Integer padding, Integer stride, Integer filterSize, ActivationFunctionType activationFunctionType) {
        Optional<Layer> layer;
        switch (layerType) {
            case CONVOLUTION:
                layer = Optional.of(new ConvolutionLayer(activationFunctionType, initializer, filterSize, padding, stride, channelCount));
                inputWidth = computeNewDimension(padding, stride, filterSize, inputWidth);
                inputHeight = computeNewDimension(padding, stride, filterSize, inputHeight);
                inputChannelCount = channelCount;
                break;
            case POOLING_MAX:
                layer = Optional.of(new MaxPoolingLayer(activationFunctionType, filterSize, padding, stride));
                inputWidth = computeNewDimension(padding, stride, filterSize, inputWidth);
                inputHeight = computeNewDimension(padding, stride, filterSize, inputHeight);
                break;
            case POOLING_AVERAGE:
                layer = Optional.of(new AveragePoolingLayer(activationFunctionType, filterSize, padding, stride));
                inputWidth = computeNewDimension(padding, stride, filterSize, inputWidth);
                inputHeight = computeNewDimension(padding, stride, filterSize, inputHeight);
                break;
            case FULLY_CONNECTED:
                layer = Optional.empty();
                break;
            default:
                throw new IllegalArgumentException(layerType + " is not allowed for convolution network");
        }
        return layer;
    }

    private int computeNewDimension(Integer padding, Integer stride, Integer filterSize, int input) {
        return (input - 2 * padding + filterSize) / stride + 1;
    }

    private void checkInputs() {
        if (inputWidth <= 0 || inputHeight <= 0) {
            throw new IllegalArgumentException("Size of input data should be set");
        }
        if (layerTypes.isEmpty()) {
            throw new IllegalArgumentException("At least one hidden layer should be set");
        }
    }

}
