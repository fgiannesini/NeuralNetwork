package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithDerivationAndRmsStopProcessProvider implements IGradientDescentWithDerivationProcessProvider {
    private final IGradientDescentWithDerivationProcessProvider processProvider;
    private final List<WeightBiasLayer> rmsStopLayers;
    private final Double rmsStopCoeff;
    private final Double epsilon;

    public GradientDescentWithDerivationAndRmsStopProcessProvider(IGradientDescentWithDerivationProcessProvider processProvider, Double rmsStopCoeff) {
        this.rmsStopCoeff = rmsStopCoeff;
        this.epsilon = Math.pow(10, -8);
        this.processProvider = processProvider;
        rmsStopLayers = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<WeightBiasLayer> layers = correctedNeuralNetworkModel.getLayers();
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initRmsStopLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                WeightBiasLayer layer = layers.get(layerIndex);
                WeightBiasLayer rmsStopLayer = rmsStopLayers.get(layerIndex);

                //Sdw = c * Sdw + (1 - c)*dW²
                rmsStopLayer.setWeightMatrix(rmsStopLayer.getWeightMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getWeightCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //W = W - a * dW/(sqrt(Sdw) + e)
                DoubleMatrix weightCorrection = gradientDescentCorrection.getWeightCorrectionResults().div(MatrixFunctions.sqrt(rmsStopLayer.getWeightMatrix()).addi(epsilon)).muli(container.getLearningRate());
                layer.getWeightMatrix().subi(weightCorrection);

                //Sdb = c *Sdb + (1 - c)*dB²
                rmsStopLayer.setBiasMatrix(rmsStopLayer.getBiasMatrix().muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getBiasCorrectionResults(), 2d).muli(1d - rmsStopCoeff)));
                //B = B - a * dB/(sqrt(Sdb) + e)
                DoubleMatrix biasCorrection = gradientDescentCorrection.getBiasCorrectionResults().div(MatrixFunctions.sqrt(rmsStopLayer.getBiasMatrix()).addi(epsilon)).muli(container.getLearningRate());
                layer.getBiasMatrix().subi(biasCorrection);
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<WeightBiasLayer> initRmsStopLayers(List<WeightBiasLayer> layers) {
        return layers.stream()
                .map(layer -> new WeightBiasLayer(layer.getInputLayerSize(), layer.getOutputLayerSize(), InitializerType.ZEROS.getInitializer(), ActivationFunctionType.NONE))
                .collect(Collectors.toList());
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher() {
        return processProvider.getGradientWithDerivationLauncher();
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return processProvider.getCostComputerBuildingLauncher();
    }
}
