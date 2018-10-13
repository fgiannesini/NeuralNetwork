package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class MaxPoolingLayer extends Layer {

    private final int filterSize;
    private final int padding;
    private final int stride;
    private final int channelCount;
    private final int inputWidth;
    private final int inputHeight;
    private final int outputWidth;
    private final int outputHeight;

    public MaxPoolingLayer(ActivationFunctionType activationFunctionType, int filterSize, int padding, int stride, int channelCount, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
        super(activationFunctionType);
        this.filterSize = filterSize;
        this.padding = padding;
        this.stride = stride;
        this.channelCount = channelCount;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
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
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MaxPoolingLayer that = (MaxPoolingLayer) o;
        return filterSize == that.filterSize &&
                padding == that.padding &&
                stride == that.stride;
    }

    @Override
    public int hashCode() {
        return Objects.hash(filterSize, padding, stride);
    }

    @Override
    public String toString() {
        return "MaxPoolingLayer{" +
                "filterSize=" + filterSize +
                ", padding=" + padding +
                ", stride=" + stride +
                '}';
    }

    @Override
    public MaxPoolingLayer clone() {
        return (MaxPoolingLayer) super.clone();
    }

    @Override
    public List<DoubleMatrix> getParametersMatrix() {
        return Collections.emptyList();
    }

    @Override
    public void accept(LayerVisitor layerVisitor) {
        layerVisitor.visit(this);
    }
}
