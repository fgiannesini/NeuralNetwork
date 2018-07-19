package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularization extends GradientDescent {

    private List<DoubleMatrix> dropOutMatrices;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;

    public GradientDescentWithDropOutRegularization(NeuralNetworkModel neuralNetworkModel, double learningRate, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        super(neuralNetworkModel, learningRate);
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
    }

    @Override
    protected IIntermediateOutputComputer buildOutputComputer(NeuralNetworkModel neuralNetworkModel) {
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .withDropOutParameters(dropOutMatrices)
                .buildIntermediateOutputComputer();
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        dropOutMatrices = dropOutMatricesSupplier.get();
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        return super.learn(inputMatrix, dropOutOutput);
    }

    @Override
    protected DoubleMatrix computeError(GradientLayerProvider provider, DoubleMatrix previousError) {
        DoubleMatrix dropOutMatrix = dropOutMatrices.get(provider.getCurrentLayerIndex());
        return super.computeError(provider, previousError).muliColumnVector(dropOutMatrix);
    }

    @Override
    protected DoubleMatrix computeFirstError(GradientLayerProvider provider, DoubleMatrix y) {
        DoubleMatrix dropOutMatrix = dropOutMatrices.get(provider.getCurrentLayerIndex());
        return super.computeFirstError(provider, y).muliColumnVector(dropOutMatrix);
    }
}
