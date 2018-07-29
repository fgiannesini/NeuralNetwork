package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescentWithDerivation implements LearningAlgorithm {

    private final NeuralNetworkModel originalNeuralNetworkModel;
    private final CostType costType;
    private final double learningRate;
    private final double step = 0.0001;

    public GradientDescentWithDerivation(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate) {
        this.originalNeuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.learningRate = learningRate;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        NeuralNetworkModel resultNeuralNetworkModel = originalNeuralNetworkModel.clone();

        List<Layer> layers = originalNeuralNetworkModel.getLayers();
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            Layer layer = layers.get(layerIndex);

            DoubleMatrix originalWeightMatrix = layer.getWeightMatrix();
            for (int elementIndex = 0; elementIndex < originalWeightMatrix.length; elementIndex++) {
                NeuralNetworkModel modifiedNeuralNetworkModel = originalNeuralNetworkModel.clone();
                CostComputer costComputer = buildCostComputer(modifiedNeuralNetworkModel);

                DoubleMatrix modifiedWeightMatrix = modifiedNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix();

                modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) + step);
                double superiorStepCost = costComputer.compute(inputMatrix, y);
                modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) - step);

                modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) - step);
                double inferiorStepCost = costComputer.compute(inputMatrix, y);
                modifiedWeightMatrix.put(elementIndex, modifiedWeightMatrix.get(elementIndex) + step);

                double dW = (superiorStepCost - inferiorStepCost) / (2 * step);
                double w = resultNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix().get(elementIndex);
                resultNeuralNetworkModel.getLayers().get(layerIndex).getWeightMatrix().put(elementIndex, w - learningRate * dW);
            }

            DoubleMatrix originalBiasMatrix = layer.getBiasMatrix();
            for (int elementIndex = 0; elementIndex < originalBiasMatrix.length; elementIndex++) {
                NeuralNetworkModel modifiedNeuralNetworkModel = originalNeuralNetworkModel.clone();
                CostComputer costComputer = buildCostComputer(modifiedNeuralNetworkModel);
                DoubleMatrix modifiedBiasMatrix = modifiedNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix();

                modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) + step);
                double superiorStepCost = costComputer.compute(inputMatrix, y);
                modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) - step);

                modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) - step);
                double inferiorStepCost = costComputer.compute(inputMatrix, y);
                modifiedBiasMatrix.put(elementIndex, modifiedBiasMatrix.get(elementIndex) + step);

                double dB = (superiorStepCost - inferiorStepCost) / (2 * step);
                double b = resultNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix().get(elementIndex);
                resultNeuralNetworkModel.getLayers().get(layerIndex).getBiasMatrix().put(elementIndex, b - learningRate * dB);
            }

        }

        return resultNeuralNetworkModel;
    }

    protected CostComputer buildCostComputer(NeuralNetworkModel modifiedNeuralNetworkModel) {
        return CostComputerBuilder.init()
                .withNeuralNetworkModel(modifiedNeuralNetworkModel)
                .withType(costType)
                .build();
    }
}
