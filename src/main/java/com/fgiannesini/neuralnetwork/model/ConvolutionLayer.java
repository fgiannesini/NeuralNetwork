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

    private int filterSize;
    private int padding;
    private int stride;
    private int channelCount;

    public ConvolutionLayer(ActivationFunctionType activationFunctionType, Initializer initializer, int filterSize, int padding, int stride, int channelCount) {
        super(activationFunctionType);
        this.padding = padding;
        this.stride = stride;
        this.filterSize = filterSize;
        this.channelCount = channelCount;
        weightMatrices = IntStream.range(0, channelCount)
                .mapToObj(i -> initializer.initDoubleMatrix(filterSize, filterSize))
                .collect(Collectors.toList());
        biasMatrices = IntStream.range(0, channelCount)
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

    public int getChannelCount() {
        return channelCount;
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
        if (o == null || getClass() != o.getClass()) return false;
        ConvolutionLayer that = (ConvolutionLayer) o;
        return filterSize == that.filterSize &&
                padding == that.padding &&
                stride == that.stride &&
                channelCount == that.channelCount &&
                Objects.equals(weightMatrices, that.weightMatrices) &&
                Objects.equals(biasMatrices, that.biasMatrices);
    }

    @Override
    public int hashCode() {

        return Objects.hash(weightMatrices, biasMatrices, filterSize, padding, stride, channelCount);
    }

    @Override
    public String toString() {
        return "ConvolutionLayer{" +
                "weightMatrices=" + weightMatrices +
                ", biasMatrices=" + biasMatrices +
                ", filterSize=" + filterSize +
                ", padding=" + padding +
                ", stride=" + stride +
                ", channelCount=" + channelCount +
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
