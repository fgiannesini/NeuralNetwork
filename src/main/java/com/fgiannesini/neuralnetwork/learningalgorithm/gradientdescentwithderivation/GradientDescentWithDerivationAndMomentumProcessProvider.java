package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithDerivationAndMomentumProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final Double momentumCoeff;
    private final IGradientDescentWithDerivationProcessProvider processProvider;
    private final List<WeightBiasLayer> momentumLayers;

    public GradientDescentWithDerivationAndMomentumProcessProvider(IGradientDescentWithDerivationProcessProvider processProvider, Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.processProvider = processProvider;
        momentumLayers = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<WeightBiasLayer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumLayers.isEmpty()) {
                momentumLayers.addAll(initMomentumLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                WeightBiasLayer layer = layers.get(layerIndex);
                WeightBiasLayer momentumLayer = momentumLayers.get(layerIndex);

                //Vdw = m*Vdw + (1 - m)*dW
                momentumLayer.setWeightMatrix(momentumLayer.getWeightMatrix().muli(momentumCoeff).addi(gradientDescentCorrection.getWeightCorrectionResults().mul(1d - momentumCoeff)));
                layer.getWeightMatrix().subi(momentumLayer.getWeightMatrix().mul(container.getLearningRate()));

                //Vdb = m*Vdb + (1 - m)*dB
                momentumLayer.setBiasMatrix(momentumLayer.getBiasMatrix().muli(momentumCoeff).addi(gradientDescentCorrection.getBiasCorrectionResults().mul(1d - momentumCoeff)));
                layer.getBiasMatrix().subi(momentumLayer.getBiasMatrix().mul(container.getLearningRate()));
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<WeightBiasLayer> initMomentumLayers(List<WeightBiasLayer> layers) {
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
