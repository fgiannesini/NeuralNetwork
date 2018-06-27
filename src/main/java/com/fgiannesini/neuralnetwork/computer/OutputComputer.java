package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

public class OutputComputer {

  private final NeuralNetworkModel model;

  OutputComputer(NeuralNetworkModel model) {
    this.model = model;
  }

  public float[] compute(float[] input) {
    return computeOutput(new FloatMatrix(input)).toArray();
  }

  public float[][] compute(float[][] input) {
    FloatMatrix inputMatrix = new FloatMatrix(input).transpose();
    return computeOutput(inputMatrix).transpose().toArray2();
  }

  public FloatMatrix computeOutput(FloatMatrix inputMatrix) {
    FloatMatrix currentMatrix = inputMatrix.dup();
    for (Layer layer : model.getLayers()) {
      currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
    }
    return currentMatrix;
  }
}
