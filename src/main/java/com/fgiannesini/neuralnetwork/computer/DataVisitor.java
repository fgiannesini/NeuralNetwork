package com.fgiannesini.neuralnetwork.computer;

public interface DataVisitor {

    LayerTypeData visit(WeightBiasData data);

    LayerTypeData visit(BatchNormData data);
}
