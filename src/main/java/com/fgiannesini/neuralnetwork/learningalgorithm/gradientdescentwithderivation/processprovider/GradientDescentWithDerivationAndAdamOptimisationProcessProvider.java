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

public class GradientDescentWithDerivationAndAdamOptimisationProcessProvider implements IGradientDescentWithDerivationProcessProvider {
    private final IGradientDescentWithDerivationProcessProvider processProvider;
    private final Double momentumCoeff;
    private final Double rmsStopCoeff;
    private final List<List<DoubleMatrix>> momentumMatrices;
    private final List<List<DoubleMatrix>> rmsStopMatrices;
    private final Double epsilon;

    public GradientDescentWithDerivationAndAdamOptimisationProcessProvider(IGradientDescentWithDerivationProcessProvider processProvider, Double momentumCoeff, Double rmsStopCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.rmsStopCoeff = rmsStopCoeff;
        this.processProvider = processProvider;
        momentumMatrices = new ArrayList<>();
        rmsStopMatrices = new ArrayList<>();
        this.epsilon = Math.pow(10, -8);
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel<Layer> correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumMatrices.isEmpty()) {
                momentumMatrices.addAll(initLayers(layers));
            }
            if (rmsStopMatrices.isEmpty()) {
                rmsStopMatrices.addAll(initLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parameterMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> momentumLayer = momentumMatrices.get(layerIndex);
                List<DoubleMatrix> rmsStopLayer = rmsStopMatrices.get(layerIndex);

                for (int parameterIndex = 0; parameterIndex < parameterMatrices.size(); parameterIndex++) {
                    //Vdw = m*Vdw + (1 - m)*dW
                    momentumLayer.set(parameterIndex, momentumLayer.get(parameterIndex).muli(momentumCoeff).addi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(1d - momentumCoeff)));
                    //Sdw = c * Sdw + (1 - c)*dW²
                    rmsStopLayer.set(parameterIndex, rmsStopLayer.get(parameterIndex).muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getCorrectionResults().get(parameterIndex), 2d).muli(1d - rmsStopCoeff)));
                    //W = W - a * Vdw/(1-m)/(sqrt(Sdw/(1-c)) + e)
                    DoubleMatrix weightCorrection = momentumLayer.get(parameterIndex).div(1d - momentumCoeff).divi(MatrixFunctions.sqrt(rmsStopLayer.get(parameterIndex).div(1d - rmsStopCoeff)).addi(epsilon)).muli(container.getLearningRate());
                    parameterMatrices.get(parameterIndex).subi(weightCorrection);
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