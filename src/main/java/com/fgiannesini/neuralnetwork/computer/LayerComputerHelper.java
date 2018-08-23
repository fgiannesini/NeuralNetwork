package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class LayerComputerHelper {

    public static DoubleMatrix computeAFromZ(DoubleMatrix z, WeightBiasLayer layer) {
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        return activationFunctionApplier.apply(z);
    }

    public static DoubleMatrix computeZFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        //W.X + b
        return layer.getWeightMatrix().mmul(input).addiColumnVector(layer.getBiasMatrix());
    }

    public static DoubleMatrix computeAFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        DoubleMatrix z = computeZFromInput(input, layer);
        return computeAFromZ(z, layer);
    }

}
