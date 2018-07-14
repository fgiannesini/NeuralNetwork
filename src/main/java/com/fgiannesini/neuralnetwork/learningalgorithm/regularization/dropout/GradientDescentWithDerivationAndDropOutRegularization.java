package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescentWithDerivationAndDropOutRegularization extends GradientDescentWithDerivation {

    private final List<DoubleMatrix> dropOutMatrices;
    private CostType costType;

    public GradientDescentWithDerivationAndDropOutRegularization(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, List<DoubleMatrix> dropOutMatrices) {
        super(neuralNetworkModel, costType, learningRate);
        this.costType = costType;
        this.dropOutMatrices = dropOutMatrices;
    }

    @Override
    protected CostComputer buildCostComputer(NeuralNetworkModel modifiedNeuralNetworkModel) {
        return CostComputerBuilder.init()
                .withNeuralNetworkModel(modifiedNeuralNetworkModel)
                .withDropOutRegularization(dropOutMatrices)
                .withType(costType)
                .build();
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        return super.learn(inputMatrix, dropOutOutput);
    }
}
