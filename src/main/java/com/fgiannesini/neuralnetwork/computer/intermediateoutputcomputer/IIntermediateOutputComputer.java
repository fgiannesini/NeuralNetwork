package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;

import java.util.List;

public interface IIntermediateOutputComputer {

    List<IntermediateOutputResult> compute(LayerTypeData data);
}
