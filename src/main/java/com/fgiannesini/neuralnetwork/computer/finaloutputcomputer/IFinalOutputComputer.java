package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;

public interface IFinalOutputComputer {

    LayerTypeData compute(LayerTypeData input);
}
