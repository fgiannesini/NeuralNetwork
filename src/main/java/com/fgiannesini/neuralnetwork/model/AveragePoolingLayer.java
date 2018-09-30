package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class AveragePoolingLayer extends Layer {
    private final int filterSize;
    private final int padding;
    private final int stride;

    public AveragePoolingLayer(ActivationFunctionType activationFunctionType, int filterSize, int padding, int stride) {
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
        if (!(o instanceof AveragePoolingLayer)) return false;
        AveragePoolingLayer that = (AveragePoolingLayer) o;
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
        return "AveragePoolingLayer{" +
                "filterSize=" + filterSize +
                ", padding=" + padding +
                ", stride=" + stride +
                '}';
    }

    @Override
    public AveragePoolingLayer clone() {
        return (AveragePoolingLayer) super.clone();
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
