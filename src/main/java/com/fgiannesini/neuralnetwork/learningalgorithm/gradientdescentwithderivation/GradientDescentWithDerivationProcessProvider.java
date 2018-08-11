package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class GradientDescentWithDerivationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final double step = 0.0001;

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher() {
        return container -> {
            DoubleMatrix output = container.getY();
            DoubleMatrix input = container.getInput();
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            List<GradientDescentCorrection> corrections = new ArrayList<>();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                Layer layer = layers.get(layerIndex);

                DoubleMatrix correctedWeightMatrix = DoubleMatrix.zeros(layer.getWeightMatrix().getRows(), layer.getWeightMatrix().getColumns());

                for (int elementIndex = 0; elementIndex < correctedWeightMatrix.length; elementIndex++) {
                    NeuralNetworkModel modifiedNeuralNetworkModel = container.getNeuralNetworkModel().clone();
                    CostComputer costComputer = container.getCostComputerProcessLauncher().apply(new GradientDescentWithDerivationCostComputerContainer(modifiedNeuralNetworkModel, container.getCostType()));

                    DoubleMatrix modifiedWeightMatrix = modifiedNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix();

                    modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) + step);
                    double superiorStepCost = costComputer.compute(input, output);
                    modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) - step);

                    modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) - step);
                    double inferiorStepCost = costComputer.compute(input, output);
                    modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) + step);

                    double dW = (superiorStepCost - inferiorStepCost) / (2 * step);
                    correctedWeightMatrix.put(elementIndex, dW);
                }

                DoubleMatrix correctedBiasMatrix = DoubleMatrix.zeros(layer.getBiasMatrix().getRows(), layer.getBiasMatrix().getColumns());
                for (int elementIndex = 0; elementIndex < correctedBiasMatrix.length; elementIndex++) {
                    NeuralNetworkModel modifiedNeuralNetworkModel = container.getNeuralNetworkModel().clone();
                    CostComputer costComputer = container.getCostComputerProcessLauncher().apply(new GradientDescentWithDerivationCostComputerContainer(modifiedNeuralNetworkModel, container.getCostType()));
                    DoubleMatrix modifiedBiasMatrix = modifiedNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix();

                    modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) + step);
                    double superiorStepCost = costComputer.compute(input, output);
                    modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) - step);

                    modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) - step);
                    double inferiorStepCost = costComputer.compute(input, output);
                    modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) + step);

                    double dB = (superiorStepCost - inferiorStepCost) / (2 * step);
                    correctedBiasMatrix.put(elementIndex, dB);
                }

                GradientDescentCorrection gradientDescentCorrection = new GradientDescentCorrection(correctedWeightMatrix, correctedBiasMatrix);
                corrections.add(gradientDescentCorrection);
            }

            return corrections;
        };
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return container -> CostComputerBuilder.init()
                .withNeuralNetworkModel(container.getNeuralNetworkModel())
                .withType(container.getCostType())
                .build();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                layer.getWeightMatrix().subi(gradientDescentCorrection.getWeightCorrectionResults().mul(container.getLearningRate()));
                layer.getBiasMatrix().subi(gradientDescentCorrection.getBiasCorrectionResults().mul(container.getLearningRate()));
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }
}
