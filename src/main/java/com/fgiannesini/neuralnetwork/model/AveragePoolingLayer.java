package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class AveragePoolingLayer extends Layer {
    private final Integer filterSize;
    private final Integer padding;
    private final Integer stride;

    public AveragePoolingLayer(ActivationFunctionType activationFunctionType, Integer filterSize, Integer padding, Integer stride) {
        super(activationFunctionType);
        this.filterSize = filterSize;
        this.padding = padding;
        this.stride = stride;
    }

    public Integer getFilterSize() {
        return filterSize;
    }

    public Integer getPadding() {
        return padding;
    }

    public Integer getStride() {
        return stride;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AveragePoolingLayer that = (AveragePoolingLayer) o;
        return Objects.equals(filterSize, that.filterSize) &&
                Objects.equals(padding, that.padding) &&
                Objects.equals(stride, that.stride);
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
