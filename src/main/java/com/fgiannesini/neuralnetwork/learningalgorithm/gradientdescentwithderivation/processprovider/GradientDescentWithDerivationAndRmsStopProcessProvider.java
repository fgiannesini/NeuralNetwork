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
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithDerivationAndRmsStopProcessProvider implements IGradientDescentWithDerivationProcessProvider {
    private final IGradientDescentWithDerivationProcessProvider processProvider;
    private final List<List<DoubleMatrix>> rmsStopLayers;
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
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initRmsStopLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> rmsStopMatrices = rmsStopLayers.get(layerIndex);

                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    //Sdw = c * Sdw + (1 - c)*dW²
                    rmsStopMatrices.set(parameterIndex, rmsStopMatrices.get(parameterIndex).muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getCorrectionResults().get(parameterIndex), 2d).muli(1d - rmsStopCoeff)));
                    //W = W - a * dW/(sqrt(Sdw) + e)
                    DoubleMatrix parameterCorrection = gradientDescentCorrection.getCorrectionResults().get(parameterIndex).div(MatrixFunctions.sqrt(rmsStopMatrices.get(parameterIndex)).addi(epsilon)).muli(container.getLearningRate());
                    parametersMatrices.get(parameterIndex).subi(parameterCorrection);
                }
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<List<DoubleMatrix>> initRmsStopLayers(List<Layer> layers) {
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
