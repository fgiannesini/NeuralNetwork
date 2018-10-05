package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.Layer;

public interface CostComputer<L extends Layer> {
    double compute(LayerTypeData input, LayerTypeData output);
}
