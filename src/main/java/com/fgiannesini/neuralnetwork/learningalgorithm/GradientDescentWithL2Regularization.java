package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.List;

public class GradientDescentWithL2Regularization extends GradientDescent {
    private final double learningRate;
    private final double regularizationCoeff;
    private final NeuralNetworkModel originalNeuralNetworkModel;

    GradientDescentWithL2Regularization(NeuralNetworkModel originalNeuralNetworkModel, double learningRate, double regularizationCoeff) {
        super(originalNeuralNetworkModel, learningRate);
        this.originalNeuralNetworkModel = originalNeuralNetworkModel;
        this.learningRate = learningRate;
        this.regularizationCoeff = regularizationCoeff;
    }

    @Override
    protected NeuralNetworkModel applyGradientDescentCorrections(List<GradientDescentCorrection> gradientDescentCorrections, int inputCount) {
        NeuralNetworkModel correctedNeuralNetworkModel = super.applyGradientDescentCorrections(gradientDescentCorrections, inputCount);
        List<Layer> layers = correctedNeuralNetworkModel.getLayers();
        List<Layer> originalLayers = originalNeuralNetworkModel.getLayers();
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            Layer layer = layers.get(layerIndex);
            Layer originalLayer = originalLayers.get(layerIndex);
            layer.getWeightMatrix().subi(originalLayer.getWeightMatrix().mul(learningRate * regularizationCoeff / inputCount));
        }
        return correctedNeuralNetworkModel;
    }
}
