package com.fgiannesini.neuralnetwork.model;

public interface LayerVisitor {

    void visit(WeightBiasLayer layer);

    void visit(BatchNormLayer layer);

    void visit(AveragePoolingLayer layer);

    void visit(MaxPoolingLayer layer);

    void visit(ConvolutionLayer layer);
}
