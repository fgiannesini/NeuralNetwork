package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithMomentumProcessProvider implements IGradientDescentProcessProvider {
    private final Double momentumCoeff;
    private final IGradientDescentProcessProvider processProvider;
    private final List<WeightBiasLayer> momentumLayers;

    public GradientDescentWithMomentumProcessProvider(IGradientDescentProcessProvider processProvider, Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.processProvider = processProvider;
        momentumLayers = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
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
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<WeightBiasLayer> initMomentumLayers(List<WeightBiasLayer> layers) {
        return layers.stream()
                .map(layer -> new WeightBiasLayer(layer.getInputLayerSize(), layer.getOutputLayerSize(), InitializerType.ZEROS.getInitializer(), ActivationFunctionType.NONE))
                .collect(Collectors.toList());
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return processProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return processProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return processProvider.getFirstErrorComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return processProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }
}
