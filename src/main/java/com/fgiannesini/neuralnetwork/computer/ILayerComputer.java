package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

public interface ILayerComputer<L extends Layer> {

    default DoubleMatrix computeAFromZ(DoubleMatrix z, L layer) {
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        return activationFunctionApplier.apply(z);
    }

    default DoubleMatrix computeAFromInput(DoubleMatrix input, L layer) {
        DoubleMatrix z = computeZFromInput(input, layer);
        return computeAFromZ(z, layer);
    }

    DoubleMatrix computeZFromInput(DoubleMatrix input, L layer);

}
