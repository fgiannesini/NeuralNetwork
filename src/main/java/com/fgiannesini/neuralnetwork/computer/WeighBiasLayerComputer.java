package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class WeighBiasLayerComputer implements ILayerComputer<WeightBiasLayer> {

    public DoubleMatrix computeZFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        //W.X + b
        return layer.getWeightMatrix().mmul(input).addiColumnVector(layer.getBiasMatrix());
    }
}
