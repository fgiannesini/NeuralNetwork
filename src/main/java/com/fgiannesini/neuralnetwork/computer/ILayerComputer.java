package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

public interface ILayerComputer<L extends Layer> {

    default DoubleMatrix computeAFromZ(DoubleMatrix z, Layer layer) {
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        return activationFunctionApplier.apply(z);
    }

    default IntermediateOutputResult computeAFromInput(DoubleMatrix input, Layer layer) {
        IntermediateOutputResult result = computeZFromInput(input, layer);
        DoubleMatrix activatedResult = computeAFromZ(result.getResult(), layer);
        result.setResult(activatedResult);
        return result;
    }

    IntermediateOutputResult computeZFromInput(DoubleMatrix input, Layer layer);

}
