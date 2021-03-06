package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrectionsContainer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithMomentumProcessProvider implements IGradientDescentProcessProvider {
    private final Double momentumCoeff;
    private final IGradientDescentProcessProvider processProvider;
    private final List<List<DoubleMatrix>> momentumLayers;

    public GradientDescentWithMomentumProcessProvider(IGradientDescentProcessProvider processProvider, Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.processProvider = processProvider;
        momentumLayers = new ArrayList<>();
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return processProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumLayers.isEmpty()) {
                momentumLayers.addAll(initMomentumLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> momentumLayer = momentumLayers.get(layerIndex);

                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    //Vdw = m*Vdw + (1 - m)*dW
                    momentumLayer.set(parameterIndex, momentumLayer.get(parameterIndex).muli(momentumCoeff).addi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(1d - momentumCoeff)));
                    parametersMatrices.get(parameterIndex).subi(momentumLayer.get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<List<DoubleMatrix>> initMomentumLayers(List<Layer> layers) {
        return layers.stream()
                .map(layer -> layer.getParametersMatrix()
                        .stream()
                        .map(p -> DoubleMatrix.zeros(p.getRows(), p.getColumns()))
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

}
