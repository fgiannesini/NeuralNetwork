package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentWithDerivationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final double step = 0.0001;

    @Override
    public Function<GradientDescentWithDerivationContainer, GradientDescentWithDerivationContainer> getGradientWithDerivationLauncher() {
        return container -> {
            NeuralNetworkModel resultNeuralNetworkModel = container.getNeuralNetworkModel().clone();
            DoubleMatrix output = container.getY();
            DoubleMatrix input = container.getInput();
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                Layer layer = layers.get(layerIndex);

                DoubleMatrix originalWeightMatrix = layer.getWeightMatrix();

                for (int elementIndex = 0; elementIndex < originalWeightMatrix.length; elementIndex++) {
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
                    double w = resultNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix().get(elementIndex);
                    resultNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix().put(elementIndex, w - container.getLearningRate() * dW);
                }

                DoubleMatrix originalBiasMatrix = layer.getBiasMatrix();
                for (int elementIndex = 0; elementIndex < originalBiasMatrix.length; elementIndex++) {
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
                    double b = resultNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix().get(elementIndex);
                    resultNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix().put(elementIndex, b - container.getLearningRate() * dB);
                }

            }

            return new GradientDescentWithDerivationContainer(input, output, resultNeuralNetworkModel, container.getLearningRate(), container.getCostType(), container.getCostComputerProcessLauncher());
        };
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return container -> CostComputerBuilder.init()
                .withNeuralNetworkModel(container.getNeuralNetworkModel())
                .withType(container.getCostType())
                .build();
    }
}
