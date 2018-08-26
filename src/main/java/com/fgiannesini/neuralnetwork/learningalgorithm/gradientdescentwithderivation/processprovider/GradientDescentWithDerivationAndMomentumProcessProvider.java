package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithDerivationAndMomentumProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final Double momentumCoeff;
    private final IGradientDescentWithDerivationProcessProvider processProvider;
    private final List<List<DoubleMatrix>> momentumMatrices;

    public GradientDescentWithDerivationAndMomentumProcessProvider(IGradientDescentWithDerivationProcessProvider processProvider, Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.processProvider = processProvider;
        momentumMatrices = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel<Layer> correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumMatrices.isEmpty()) {
                momentumMatrices.addAll(initLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> momentumMatrices = this.momentumMatrices.get(layerIndex);

                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    //Vdw = m*Vdw + (1 - m)*dW
                    momentumMatrices.set(parameterIndex, momentumMatrices.get(parameterIndex).muli(momentumCoeff).addi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(1d - momentumCoeff)));
                    parametersMatrices.get(parameterIndex).subi(momentumMatrices.get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<List<DoubleMatrix>> initLayers(List<Layer> layers) {
        return layers.stream()
                .map(layer -> layer.getParametersMatrix()
                        .stream()
                        .map(p -> DoubleMatrix.zeros(p.getRows(), p.getColumns()))
                        .collect(Collectors.toList()))
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
