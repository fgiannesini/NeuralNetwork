package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Supplier;

public class GradientDescentWithDerivationAndDropOutRegularization extends GradientDescentWithDerivation {

    private List<DoubleMatrix> dropOutMatrices;
    private final CostType costType;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;

    public GradientDescentWithDerivationAndDropOutRegularization(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        super(neuralNetworkModel, costType, learningRate);
        this.costType = costType;
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
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
        dropOutMatrices = dropOutMatricesSupplier.get();
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        return super.learn(inputMatrix, dropOutOutput);
    }
}
