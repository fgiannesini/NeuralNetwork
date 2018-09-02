package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class WeighBiasLayerComputer implements ILayerComputer<WeightBiasLayer> {

    public IntermediateOutputResult computeZFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        //W.X + b
        DoubleMatrix result = layer.getWeightMatrix().mmul(input).addiColumnVector(layer.getBiasMatrix());
        return new IntermediateOutputResult(result);
    }
}
