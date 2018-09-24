package com.fgiannesini.neuralnetwork.computer;

public interface DataVisitor {

    void visit(WeightBiasData data);

    void visit(BatchNormData data);
}
