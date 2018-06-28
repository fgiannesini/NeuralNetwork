package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.FloatMatrix;

public class LayerComputerHelper {

  public static FloatMatrix computeAFromZ(FloatMatrix z, Layer layer) {
    ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
    return activationFunctionApplier.apply(z);
  }

  public static FloatMatrix computeZFromInput(FloatMatrix input, Layer layer) {
    //Wt.X + b
    return layer.getWeightMatrix().transpose().mmul(input).addiColumnVector(layer.getBiasMatrix());
  }

  public static FloatMatrix computeAFromInput(FloatMatrix input, Layer layer) {
    FloatMatrix z = computeZFromInput(input,layer);
    return computeAFromZ(z,layer);
  }

}
