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
  private NeuralNetworkModel neuralNetworkModel;
    private double learningRate;

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
        DoubleMatrix dz = currentGradientDescentLayerResult.getaLayerResults().sub(y); //dZ2 = A2 - Y
        DoubleMatrix weightCorrection = nextGradientDescentLayerResult.getaLayerResults().mmul(dz.transpose()).divi(inputCount); //dW2 = 1/m * A1 * dZ2t
        DoubleMatrix biasCorrection = dz.rowSums().divi(inputCount); //dB2 = 1/m * sum(dZ2)

    gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

    for (int layerIndex = 1; layerIndex < reverseGradientDescentLayerData.size() - 1; layerIndex++) {
        GradientDescentLayerResult previousGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex - 1);
      currentGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex);
      nextGradientDescentLayerResult = reverseGradientDescentLayerData.get(layerIndex + 1);
      ActivationFunctionApplier activationFunction = currentGradientDescentLayerResult.getActivationFunctionType().getActivationFunction();
        dz = previousGradientDescentLayerResult.getWeightMatrix().mmul(dz)
                .muli(activationFunction.derivate(currentGradientDescentLayerResult.getzLayerResults())); //dZ1 = W2 * dZ2 .* g1'(Z1)
        weightCorrection = nextGradientDescentLayerResult.getaLayerResults().mmul(dz.transpose()).divi(inputCount); //dW1 = 1/m * A0 * dZ1t
      biasCorrection = dz.rowSums().divi(inputCount); //dB1 = 1/m * sum(dZ1)
      gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
    }

    Collections.reverse(gradientDescentCorrections);

    return gradientDescentCorrections;
  }

    private List<GradientDescentLayerResult> buildReverseGradientDescentLayerResults(List<GradientDescentLayerResult> gradientDescentLayerData, DoubleMatrix inputMatrix) {
        List<GradientDescentLayerResult> reverseGradientDescentLayerData = new ArrayList<>(gradientDescentLayerData);
        Collections.reverse(reverseGradientDescentLayerData);
        GradientDescentLayerResult inputResult = new GradientDescentLayerResult();
        inputResult.setAResultLayer(inputMatrix);
        inputResult.setZResultLayer(inputMatrix);
        reverseGradientDescentLayerData.add(inputResult);
        return reverseGradientDescentLayerData;
    }

    private List<GradientDescentLayerResult> launchForwardComputation(DoubleMatrix inputMatrix) {
    List<Layer> layers = neuralNetworkModel.getLayers();
    List<GradientDescentLayerResult> gradientDescentLayerResults = new ArrayList<>();
        DoubleMatrix currentResult = inputMatrix;
    for (Layer layer : layers) {
        GradientDescentLayerResult gradientDescentLayerResult = new GradientDescentLayerResult(layer.getWeightMatrix(), layer.getActivationFunctionType());
        DoubleMatrix zResult = LayerComputerHelper.computeZFromInput(currentResult, layer);
      gradientDescentLayerResult.setZResultLayer(zResult);
        DoubleMatrix aResult = LayerComputerHelper.computeAFromZ(zResult, layer);
      gradientDescentLayerResult.setAResultLayer(aResult);
      gradientDescentLayerResults.add(gradientDescentLayerResult);
      currentResult = aResult;
    }
    return gradientDescentLayerResults;
  }

}
