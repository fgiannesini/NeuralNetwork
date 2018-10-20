package com.fgiannesini.neuralnetwork.computer.data;

public interface LayerTypeData {

    void accept(DataVisitor dataVisitor);

    int getInputCount();
}
