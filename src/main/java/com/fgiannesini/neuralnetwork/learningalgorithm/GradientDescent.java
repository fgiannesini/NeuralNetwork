package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private final NeuralNetworkModel correctedNeuralNetworkModel;
    private final double learningRate;

    public GradientDescent(NeuralNetworkModel originalNeuralNetworkModel, double learningRate) {
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

        DoubleMatrix dz = computeFirstError(provider, y);
        DoubleMatrix weightCorrection = computeWeightCorrection(provider.getPreviousResult(), dz, inputCount);
        DoubleMatrix biasCorrection = computeBiasCorrection(dz, inputCount);

        gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));

        for (provider.nextLayer(); provider.hasNextLayer(); provider.nextLayer()) {
            dz = computeError(provider, dz);
            weightCorrection = computeWeightCorrection(provider.getPreviousResult(), dz, inputCount);
            biasCorrection = computeBiasCorrection(dz, inputCount);
            gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
        }

        Collections.reverse(gradientDescentCorrections);

        return gradientDescentCorrections;
    }

    protected DoubleMatrix computeError(GradientLayerProvider provider, DoubleMatrix previousError) {
        //dZ1 = W2t * dZ2 .* g1'(A1)
        previousError = provider.getPreviousWeightMatrix().transpose()
                .mmul(previousError)
                .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
        return previousError;
    }

    protected DoubleMatrix computeFirstError(GradientLayerProvider provider, DoubleMatrix y) {
        //dZ2 = (A2 - Y) .* g2'(A2)
        return provider.getCurrentResult()
                .sub(y)
                .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
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
        IIntermediateOutputComputer intermediateOutputComputer = buildOutputComputer(correctedNeuralNetworkModel);
        List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(inputMatrix);
        return new GradientLayerProvider(layers, intermediateResults);
    }

    protected IIntermediateOutputComputer buildOutputComputer(NeuralNetworkModel neuralNetworkModel) {
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildIntermediateOutputComputer();
    }

}
