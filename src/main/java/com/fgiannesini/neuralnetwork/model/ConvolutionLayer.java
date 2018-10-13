package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.Initializer;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ConvolutionLayer extends Layer {

    private List<DoubleMatrix> weightMatrices;
    private List<DoubleMatrix> biasMatrices;

    private final int filterSize;
    private final int padding;
    private final int stride;
    private final int inputChannelCount;
    private final int inputWidth;
    private final int inputHeight;
    private final int outputWidth;
    private final int outputHeight;
    private final int outputChannelCount;

    public ConvolutionLayer(ActivationFunctionType activationFunctionType, Initializer initializer, int filterSize, int padding, int stride, int outputChannelCount, int inputChannelCount, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
        super(activationFunctionType);
        this.padding = padding;
        this.stride = stride;
        this.filterSize = filterSize;
        this.outputChannelCount = outputChannelCount;
        this.inputChannelCount = inputChannelCount;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        weightMatrices = IntStream.range(0, outputChannelCount * this.inputChannelCount)
                .mapToObj(i -> initializer.initDoubleMatrix(filterSize, filterSize))
                .collect(Collectors.toList());
        biasMatrices = IntStream.range(0, outputChannelCount)
                .mapToObj(i -> initializer.initDoubleMatrix(1, 1))
                .collect(Collectors.toList());
    }

    public List<DoubleMatrix> getWeightMatrices() {
        return weightMatrices;
    }

    public List<DoubleMatrix> getBiasMatrices() {
        return biasMatrices;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public int getPadding() {
        return padding;
    }

    public int getStride() {
        return stride;
    }

    public int getOutputChannelCount() {
        return outputChannelCount;
    }

    public int getInputChannelCount() {
        return inputChannelCount;
    }


    public int getOutputWidth() {
        return outputWidth;
    }

    public int getOutputHeight() {
        return outputHeight;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    @Override
    public ConvolutionLayer clone() {
        ConvolutionLayer clone = (ConvolutionLayer) super.clone();
        clone.weightMatrices = weightMatrices.stream().map(DoubleMatrix::dup).collect(Collectors.toList());
        clone.biasMatrices = biasMatrices.stream().map(DoubleMatrix::dup).collect(Collectors.toList());
        return clone;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ConvolutionLayer)) return false;
        ConvolutionLayer that = (ConvolutionLayer) o;
        return inputChannelCount == that.inputChannelCount &&
                filterSize == that.filterSize &&
                padding == that.padding &&
                stride == that.stride &&
                outputChannelCount == that.outputChannelCount &&
                Objects.equals(weightMatrices, that.weightMatrices) &&
                Objects.equals(biasMatrices, that.biasMatrices);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputChannelCount, weightMatrices, biasMatrices, filterSize, padding, stride, outputChannelCount);
    }

    @Override
    public String toString() {
        return "ConvolutionLayer{" +
                "weightMatrices=" + weightMatrices +
                ", biasMatrices=" + biasMatrices +
                ", filterSize=" + filterSize +
                ", padding=" + padding +
                ", stride=" + stride +
                ", inputChannelCount=" + inputChannelCount +
                ", outputChannelCount=" + outputChannelCount +
                '}';
    }

    @Override
    public List<DoubleMatrix> getParametersMatrix() {
        return Stream.of(weightMatrices, biasMatrices).flatMap(List::stream).collect(Collectors.toList());
    }

    @Override
    public void accept(LayerVisitor layerVisitor) {
        layerVisitor.visit(this);
    }
}
