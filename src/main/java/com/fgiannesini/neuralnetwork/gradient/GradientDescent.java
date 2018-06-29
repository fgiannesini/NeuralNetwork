package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescent implements LearningAlgorithm {
  private NeuralNetworkModel neuralNetworkModel;
  private float learningRate;

  GradientDescent(NeuralNetworkModel neuralNetworkModel, float learningRate) {
    this.neuralNetworkModel = neuralNetworkModel.clone();
    this.learningRate = learningRate;
  }

  @Override
  public NeuralNetworkModel learn(FloatMatrix inputMatrix, FloatMatrix y) {
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

  private List<GradientDescentCorrection> launchBackwardComputation(List<GradientDescentLayerResult> gradientDescentLayerData, FloatMatrix y,
                                                                    FloatMatrix inputMatrix) {
    List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>(gradientDescentLayerData.size());
    List<GradientDescentLayerResult> reverseGradientDescentLayerData = new ArrayList<>(gradientDescentLayerData);
    Collections.reverse(reverseGradientDescentLayerData);
    int inputCount = inputMatrix.getColumns();

    GradientDescentLayerResult currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(0);
    GradientDescentLayerResult nextGradientDescentLayerResult = reverseGradientDescentLayerData.get(1);
    FloatMatrix dz = currentGradientDescentLayerResult.getaLayerResults().sub(y); //dZ2 = A2 - Y
    FloatMatrix weightCorrection = dz.mmul(nextGradientDescentLayerResult.getaLayerResults().transpose()).divi(inputCount); //dW2 = 1/m * dZ2 * A1t
    FloatMatrix biasCorrection = dz.rowSums().divi(inputCount); //dB2 = 1/m * sum(dZ2)

    gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

    for (int layerIndex = 1; layerIndex < reverseGradientDescentLayerData.size() - 1; layerIndex++) {
      currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex);
      nextGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex + 1);
      ActivationFunctionApplier activationFunction = currentGradientDescentLayerResult.getActivationFunctionType().getActivationFunction();
      dz = weightCorrection.transpose().mmul(dz).muli(
        activationFunction.derivate(currentGradientDescentLayerResult.getzLayerResults())); //dZ1 = W2t * dZ2 .* g1'(Z1)
      weightCorrection = dz.mmul(nextGradientDescentLayerResult.getaLayerResults().transpose()).divi(inputCount); //dW1 = 1/m * dZ1 * A0t
      biasCorrection = dz.rowSums().divi(inputCount); //dB1 = 1/m * sum(dZ1)
      gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
    }

    currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(reverseGradientDescentLayerData.size() - 1);
    ActivationFunctionApplier activationFunction = currentGradientDescentLayerResult.getActivationFunctionType().getActivationFunction();
    dz = weightCorrection.transpose().mmul(dz).muli(
      activationFunction.derivate(currentGradientDescentLayerResult.getzLayerResults()));//dZ0 = W1t * dZ1 .* g0'(Z0)
    weightCorrection = dz.mmul(inputMatrix.transpose()).divi(inputCount); //dW0 = 1/m * dZ0 * X
    biasCorrection = dz.rowSums().divi(inputCount);//dB0 = 1/m * sum(dZ0)
    gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

    Collections.reverse(gradientDescentCorrections);

    return gradientDescentCorrections;
  }

  private List<GradientDescentLayerResult> launchForwardComputation(FloatMatrix inputMatrix) {
    List<Layer> layers = neuralNetworkModel.getLayers();
    List<GradientDescentLayerResult> gradientDescentLayerResults = new ArrayList<>();
    FloatMatrix currentResult = inputMatrix;
    for (Layer layer : layers) {
      GradientDescentLayerResult gradientDescentLayerResult = new GradientDescentLayerResult(layer.getActivationFunctionType());
      FloatMatrix zResult = LayerComputerHelper.computeZFromInput(currentResult, layer);
      gradientDescentLayerResult.setZResultLayer(zResult);
      FloatMatrix aResult = LayerComputerHelper.computeAFromZ(zResult, layer);
      gradientDescentLayerResult.setAResultLayer(aResult);
      gradientDescentLayerResults.add(gradientDescentLayerResult);
      currentResult = aResult;
    }
    return gradientDescentLayerResults;
  }

  private static class GradientDescentLayerResult {
    private FloatMatrix zLayerResults;
    private FloatMatrix aLayerResults;

    private ActivationFunctionType activationFunctionType;

    public GradientDescentLayerResult(ActivationFunctionType activationFunctionType) {
      this.activationFunctionType = activationFunctionType;
    }

    public ActivationFunctionType getActivationFunctionType() {
      return activationFunctionType;
    }

    void setZResultLayer(FloatMatrix zLayerResult) {
      zLayerResults = zLayerResult;
    }

    void setAResultLayer(FloatMatrix aLayerResult) {
      aLayerResults = aLayerResult;
    }

    public FloatMatrix getzLayerResults() {
      return zLayerResults;
    }

    public FloatMatrix getaLayerResults() {
      return aLayerResults;
    }

  }

  private static class GradientDescentCorrection {
    private FloatMatrix weightCorrectionResults;
    private FloatMatrix biasCorrectionResults;

    public GradientDescentCorrection(FloatMatrix weightCorrectionResults, FloatMatrix biasCorrectionResults) {
      this.weightCorrectionResults = weightCorrectionResults;
      this.biasCorrectionResults = biasCorrectionResults;
    }

    public FloatMatrix getWeightCorrectionResults() {
      return weightCorrectionResults;
    }

    public FloatMatrix getBiasCorrectionResults() {
      return biasCorrectionResults;
    }
  }
}
