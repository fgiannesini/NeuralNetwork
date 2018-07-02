package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescent implements LearningAlgorithm {
  private final NeuralNetworkModel neuralNetworkModel;
  private final double learningRate;

  GradientDescent(NeuralNetworkModel neuralNetworkModel, double learningRate) {
    this.neuralNetworkModel = neuralNetworkModel.clone();
    this.learningRate = learningRate;
  }

  @Override
  public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
    List<GradientDescentLayerResult> gradientDescentLayerData = launchForwardComputation(inputMatrix);
    List<GradientDescentCorrection> gradientDescentCorrections = launchBackwardComputation(gradientDescentLayerData, y, inputMatrix);
    return applyGradientDescentCorrections(gradientDescentCorrections);
  }

  private NeuralNetworkModel applyGradientDescentCorrections(List<GradientDescentCorrection> gradientDescentCorrections) {
    List<Layer> layers = neuralNetworkModel.getLayers();
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
      GradientDescentCorrection gradientDescentCorrection = gradientDescentCorrections.get(layerIndex);
      Layer layer = layers.get(layerIndex);
      layer.getWeightMatrix().subi(gradientDescentCorrection.getWeightCorrectionResults().mul(learningRate));
      layer.getBiasMatrix().subi(gradientDescentCorrection.getBiasCorrectionResults().mul(learningRate));
    }
    return neuralNetworkModel;
  }

  private List<GradientDescentCorrection> launchBackwardComputation(List<GradientDescentLayerResult> gradientDescentLayerData, DoubleMatrix y,
                                                                    DoubleMatrix inputMatrix) {
    List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>(gradientDescentLayerData.size());
    List<GradientDescentLayerResult> reverseGradientDescentLayerData = buildReverseGradientDescentLayerResults(gradientDescentLayerData, inputMatrix);
    int inputCount = inputMatrix.getColumns();

    GradientDescentLayerResult currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(0);
    GradientDescentLayerResult nextGradientDescentLayerResult = reverseGradientDescentLayerData.get(1);
    ActivationFunctionApplier activationFunction = currentGradientDescentLayerResult.getActivationFunctionType().getActivationFunction();
    //dZ2 = (A2 - Y) .* g2'(A2)
    DoubleMatrix dz = currentGradientDescentLayerResult.getaLayerResults()
      .sub(y)
      .muli(activationFunction.derivate(currentGradientDescentLayerResult.getaLayerResults()));
    DoubleMatrix weightCorrection = computeWeightCorrection(nextGradientDescentLayerResult, dz, inputCount);
    DoubleMatrix biasCorrection = computeBiasCorrection(dz, inputCount);

    gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

    for (int layerIndex = 1; layerIndex < reverseGradientDescentLayerData.size() - 1; layerIndex++) {
      GradientDescentLayerResult previousGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex - 1);
      currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex);
      nextGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex + 1);
      activationFunction = currentGradientDescentLayerResult.getActivationFunctionType().getActivationFunction();
      //dZ1 = W2t * dZ2 .* g1'(A1)
      dz = previousGradientDescentLayerResult.getWeightMatrix().transpose()
        .mmul(dz)
        .muli(activationFunction.derivate(currentGradientDescentLayerResult.getaLayerResults()));
      weightCorrection = computeWeightCorrection(nextGradientDescentLayerResult, dz, inputCount);
      biasCorrection = computeBiasCorrection(dz, inputCount);
      gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
    }

    Collections.reverse(gradientDescentCorrections);

    return gradientDescentCorrections;
  }

  private DoubleMatrix computeBiasCorrection(DoubleMatrix dz, int inputCount) {
    //dB = sum(dZ) ./ m
    return dz.rowSums()
      .divi(inputCount);
  }

  private DoubleMatrix computeWeightCorrection(GradientDescentLayerResult nextGradientDescentLayerResult, DoubleMatrix dz, int inputCount) {
    //dW1 = dZ1 * A0t ./m
    return dz
      .mmul(nextGradientDescentLayerResult.getaLayerResults().transpose())
      .divi(inputCount);
  }

  private List<GradientDescentLayerResult> buildReverseGradientDescentLayerResults(List<GradientDescentLayerResult> gradientDescentLayerData,
                                                                                   DoubleMatrix inputMatrix) {
    List<GradientDescentLayerResult> reverseGradientDescentLayerData = new ArrayList<>(gradientDescentLayerData);
    Collections.reverse(reverseGradientDescentLayerData);
    GradientDescentLayerResult inputResult = new GradientDescentLayerResult();
    inputResult.setAResultLayer(inputMatrix);
    reverseGradientDescentLayerData.add(inputResult);
    return reverseGradientDescentLayerData;
  }

  private List<GradientDescentLayerResult> launchForwardComputation(DoubleMatrix inputMatrix) {
    List<Layer> layers = neuralNetworkModel.getLayers();
    List<GradientDescentLayerResult> gradientDescentLayerResults = new ArrayList<>();
    DoubleMatrix currentResult = inputMatrix;
    for (Layer layer : layers) {
      GradientDescentLayerResult gradientDescentLayerResult = new GradientDescentLayerResult(layer.getWeightMatrix(),
                                                                                             layer.getActivationFunctionType());
      DoubleMatrix zResult = LayerComputerHelper.computeZFromInput(currentResult, layer);
      DoubleMatrix aResult = LayerComputerHelper.computeAFromZ(zResult, layer);
      gradientDescentLayerResult.setAResultLayer(aResult);
      gradientDescentLayerResults.add(gradientDescentLayerResult);
      currentResult = aResult;
    }
    return gradientDescentLayerResults;
  }

}
