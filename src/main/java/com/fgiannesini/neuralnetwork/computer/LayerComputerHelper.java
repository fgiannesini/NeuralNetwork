package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

public class LayerComputerHelper {

  public static DoubleMatrix computeAFromZ(DoubleMatrix z, Layer layer) {
    ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
    return activationFunctionApplier.apply(z);
  }

  public static DoubleMatrix computeZFromInput(DoubleMatrix input, Layer layer) {
    //Wt.X + b
    return layer.getWeightMatrix().transpose().mmul(input).addiColumnVector(layer.getBiasMatrix());
  }

  public static DoubleMatrix computeAFromInput(DoubleMatrix input, Layer layer) {
    DoubleMatrix z = computeZFromInput(input, layer);
    return computeAFromZ(z,layer);
  }

}
