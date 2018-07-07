package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private final NeuralNetworkModel correctedNeuralNetworkModel;
    private final double learningRate;

    GradientDescent(NeuralNetworkModel originalNeuralNetworkModel, double learningRate) {
        this.correctedNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = learningRate;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        GradientLayerProvider provider = launchForwardComputation(inputMatrix);
        List<GradientDescentCorrection> gradientDescentCorrections = launchBackwardComputation(provider, y);
        return applyGradientDescentCorrections(gradientDescentCorrections, y.getColumns());
    }

    protected NeuralNetworkModel applyGradientDescentCorrections(List<GradientDescentCorrection> gradientDescentCorrections, int inputCount) {
        List<Layer> layers = correctedNeuralNetworkModel.getLayers();
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            GradientDescentCorrection gradientDescentCorrection = gradientDescentCorrections.get(layerIndex);
            Layer layer = layers.get(layerIndex);
            layer.getWeightMatrix().subi(gradientDescentCorrection.getWeightCorrectionResults().mul(learningRate));
            layer.getBiasMatrix().subi(gradientDescentCorrection.getBiasCorrectionResults().mul(learningRate));
        }
        return correctedNeuralNetworkModel;
    }

    private List<GradientDescentCorrection> launchBackwardComputation(GradientLayerProvider provider, DoubleMatrix y) {

        List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>();
        int inputCount = y.getColumns();

        //dZ2 = (A2 - Y) .* g2'(A2)
        DoubleMatrix dz = provider.getCurrentResult()
                .sub(y)
                .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
        DoubleMatrix weightCorrection = computeWeightCorrection(provider.getPreviousResult(), dz, inputCount);
        DoubleMatrix biasCorrection = computeBiasCorrection(dz, inputCount);

        gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

        for (provider.nextLayer(); provider.hasNextLayer(); provider.nextLayer()) {
            //dZ1 = W2t * dZ2 .* g1'(A1)
            dz = provider.getPreviousWeightMatrix().transpose()
                    .mmul(dz)
                    .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
            weightCorrection = computeWeightCorrection(provider.getPreviousResult(), dz, inputCount);
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

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousaLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousaLayerResult.transpose())
                .divi(inputCount);
    }

    private GradientLayerProvider launchForwardComputation(DoubleMatrix inputMatrix) {
        List<Layer> layers = correctedNeuralNetworkModel.getLayers();
        GradientLayerProvider gradientLayerProvider = new GradientLayerProvider(correctedNeuralNetworkModel.getLayers());
        gradientLayerProvider.addGradientLayerResult(inputMatrix);
        DoubleMatrix currentResult = inputMatrix;
        for (Layer layer : layers) {
            DoubleMatrix zResult = LayerComputerHelper.computeZFromInput(currentResult, layer);
            DoubleMatrix aResult = LayerComputerHelper.computeAFromZ(zResult, layer);
            gradientLayerProvider.addGradientLayerResult(aResult);
            currentResult = aResult;
        }
        return gradientLayerProvider;
    }

}
