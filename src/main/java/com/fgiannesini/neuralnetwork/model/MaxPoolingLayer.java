package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;

import java.util.Objects;

public class MaxPoolingLayer extends Layer {

    private final int filterSize;
    private final int padding;
    private final int stride;

    public MaxPoolingLayer(ActivationFunctionType activationFunctionType, int filterSize, int padding, int stride) {
        super(activationFunctionType);
        this.filterSize = filterSize;
        this.padding = padding;
        this.stride = stride;
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
}
