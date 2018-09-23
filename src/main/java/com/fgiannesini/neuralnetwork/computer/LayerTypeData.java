package com.fgiannesini.neuralnetwork.computer;

public interface LayerTypeData {

    LayerTypeData accept(DataVisitor dataVisitor);
}
