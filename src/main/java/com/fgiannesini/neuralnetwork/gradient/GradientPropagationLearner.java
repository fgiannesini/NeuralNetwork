package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.List;

public class GradientPropagationLearner {
  private NeuralNetworkModel neuralNetworkModel;

  GradientPropagationLearner(NeuralNetworkModel neuralNetworkModel) {
    this.neuralNetworkModel = neuralNetworkModel.clone();
  }

  public NeuralNetworkModel learn(float[] input, float[] expected) {
    FloatMatrix inputMatrix = new FloatMatrix(input);
    FloatMatrix y = new FloatMatrix(expected);
    return learn(inputMatrix, y);
  }

  public NeuralNetworkModel learn(float[][] input, float[][] expected) {
    FloatMatrix inputMatrix = new FloatMatrix(input);
    FloatMatrix y = new FloatMatrix(expected);
    return learn(inputMatrix, y);
  }

  NeuralNetworkModel learn(FloatMatrix inputMatrix, FloatMatrix y) {
    LayerResults layerResults = launchForwardComputation(inputMatrix);
    return launchBackwardComputation(layerResults, y, inputMatrix);
  }

  private NeuralNetworkModel launchBackwardComputation(LayerResults layerResults, FloatMatrix y, FloatMatrix inputMatrix) {
    FloatMatrix a = layerResults.getAResultLayerAtEndIndex(0);
    FloatMatrix dz = a.subi(y);
    FloatMatrix dW = dz.mmul(layerResults.getAResultLayerAtEndIndex(1).transpose());
    FloatMatrix db = dz;
    List<Layer> layers = neuralNetworkModel.getLayers();
    for (int layerIndexFormEnd = 1; layerIndexFormEnd < layers.size() - 1; layerIndexFormEnd++) {
      dz = layers.get(layerIndexFormEnd).getWeightMatrix().transpose().mmuli(dz).muli(layerResults.getZResultLayerAtEndIndex(layerIndexFormEnd));
      dW = dz.mmul(layerResults.getAResultLayerAtEndIndex(layerIndexFormEnd + 1).transpose());
    }
    dz = layers.get(0).getWeightMatrix().transpose().mmuli(dz).muli(layerResults.getZResultLayerAtEndIndex(layers.size() - 2));
    dW = dz.mmul(inputMatrix.transpose());
    db = dz;

    return neuralNetworkModel;
  }

  private LayerResults launchForwardComputation(FloatMatrix inputMatrix) {
    List<Layer> layers = neuralNetworkModel.getLayers();
    LayerResults layerResults = new LayerResults(layers.size());
    for (Layer layer : layers) {
      FloatMatrix zResult = LayerComputerHelper.computeZFromInput(inputMatrix, layer);
      layerResults.addZResultLayer(zResult);
      FloatMatrix aResult = LayerComputerHelper.computeAFromZ(zResult, layer);
      layerResults.addAResultLayer(aResult);
    }
    return layerResults;
  }

  private static class LayerResults {
    private final int lastIndexOfList;
    private List<FloatMatrix> zLayerResults;
    private List<FloatMatrix> aLayerResults;

    public LayerResults(int layerSize) {
      aLayerResults = new ArrayList<>(layerSize);
      zLayerResults = new ArrayList<>(layerSize);
      this.lastIndexOfList = layerSize - 1;
    }

    FloatMatrix getZResultLayerAtEndIndex(int layerIndexFromEnd) {
      return zLayerResults.get(lastIndexOfList - layerIndexFromEnd);
    }

    FloatMatrix getAResultLayerAtEndIndex(int layerIndexFromEnd) {
      return aLayerResults.get(lastIndexOfList - layerIndexFromEnd);
    }

    void addZResultLayer(FloatMatrix zLayerResult) {
      zLayerResults.add(zLayerResult);
    }

    void addAResultLayer(FloatMatrix aLayerResult) {
      aLayerResults.add(aLayerResult);
    }
  }
}
