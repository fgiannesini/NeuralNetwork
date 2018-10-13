package com.fgiannesini.neuralnetwork.computer.data;

public interface DataVisitor {

    void visit(WeightBiasData data);

    void visit(BatchNormData data);

    default void visit(ConvolutionData convolutionData) {
        throw new UnsupportedOperationException("Need to be implemented");
    }

    default void visit(AveragePoolingData averagePoolingData) {
        throw new UnsupportedOperationException("Need to be implemented");
    }

    default void visit(MaxPoolingData maxPoolingData) {
        throw new UnsupportedOperationException("Need to be implemented");
    }
}
