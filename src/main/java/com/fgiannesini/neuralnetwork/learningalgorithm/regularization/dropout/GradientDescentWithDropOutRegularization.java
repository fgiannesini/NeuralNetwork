package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularization implements LearningAlgorithm, InternalGradientDescent {

    private List<DoubleMatrix> dropOutMatrices;
    private InternalGradientDescent gradientDescent;
    private double learningRate;
    private NeuralNetworkModel neuralNetworkModel;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;

    public GradientDescentWithDropOutRegularization(InternalGradientDescent gradientDescent, double learningRate, NeuralNetworkModel neuralNetworkModel, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        this.gradientDescent = gradientDescent;
        this.learningRate = learningRate;
        this.neuralNetworkModel = neuralNetworkModel.clone();
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        dropOutMatrices = dropOutMatricesSupplier.get();
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        GradientLayerProvider gradientLayerProvider = getForwardComputationLauncher().apply(new ForwardComputationContainer(inputMatrix, neuralNetworkModel));
        List<GradientDescentCorrection> gradientDescentCorrections = getBackwardComputationLauncher().apply(new BackwardComputationContainer(gradientLayerProvider, dropOutOutput, getFirstErrorComputationLauncher(), getErrorComputationLauncher()));
        return getGradientDescentCorrectionsLauncher().apply(new GradientDescentCorrectionsContainer(neuralNetworkModel, gradientDescentCorrections, y.getColumns(), learningRate));
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return gradientDescent.getErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return gradientDescent.getFirstErrorComputationLauncher().andThen(container -> {
            DoubleMatrix dropOutMatrix = dropOutMatrices.get(container.getProvider().getCurrentLayerIndex());
            DoubleMatrix error = container.getPreviousError().muliColumnVector(dropOutMatrix);
            return new ErrorComputationContainer(container.getProvider(), error);
        });
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, NeuralNetworkModel> getGradientDescentCorrectionsLauncher() {
        return gradientDescent.getGradientDescentCorrectionsLauncher();
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return gradientDescent.getBackwardComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return container -> {
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(container.getNeuralNetworkModel())
                    .withDropOutParameters(dropOutMatrices)
                    .buildIntermediateOutputComputer();
            List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return new GradientLayerProvider(layers, intermediateResults);
        };
    }
}
