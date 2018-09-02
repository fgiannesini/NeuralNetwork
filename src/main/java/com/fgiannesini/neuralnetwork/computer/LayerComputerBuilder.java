package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.LayerType;

public class LayerComputerBuilder {

    private LayerType layerType;

    private LayerComputerBuilder() {
        layerType = LayerType.WEIGHT_BIAS;
    }

    public static LayerComputerBuilder init() {
        return new LayerComputerBuilder();
    }

    public LayerComputerBuilder withLayerType(LayerType layerType) {
        this.layerType = layerType;
        return this;
    }

    public <L extends Layer> ILayerComputer<L> build() {
        switch (layerType) {
            case BATCH_NORM:
                return (ILayerComputer<L>) new BatchNormLayerComputer(new MeanDeviationProvider());
            case WEIGHT_BIAS:
                return (ILayerComputer<L>) new WeighBiasLayerComputer();
            default:
                throw new IllegalArgumentException("Layer computer for type " + layerType.name() + " not implemented");
        }
    }
}
