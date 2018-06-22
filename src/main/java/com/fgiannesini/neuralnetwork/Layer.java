package com.fgiannesini.neuralnetwork;

import org.jblas.DoubleMatrix;

public class Layer {

  private int inputLayerSize;
  private int outputLayerSize;
  private DoubleMatrix weightMatrix;

  public Layer(int inputLayerSize, int outputLayerSize) {
    this.inputLayerSize = inputLayerSize;
    this.outputLayerSize = outputLayerSize;
    weightMatrix = DoubleMatrix.rand(inputLayerSize, outputLayerSize);
  }

}
