package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;

public class DataContainer {

    private final LayerTypeData input;
    private final LayerTypeData output;

    public DataContainer(LayerTypeData inputData, LayerTypeData outputData) {
        this.input = inputData;
        this.output = outputData;
    }

    public LayerTypeData getInput() {
        return input;
    }

    public LayerTypeData getOutput() {
        return output;
    }
}
