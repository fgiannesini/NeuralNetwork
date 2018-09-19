package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;

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
}
