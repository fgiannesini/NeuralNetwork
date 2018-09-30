package com.fgiannesini.neuralnetwork.computer;

public interface DataVisitor {

    void visit(WeightBiasData data);

    void visit(BatchNormData data);

    default void visit(ConvolutionData convolutionData) {
        throw new UnsupportedOperationException("Need to be implemented");
    }
}
