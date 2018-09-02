package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithRmsStopProcessProvider<L extends Layer> implements IGradientDescentProcessProvider<L> {
    private final List<List<DoubleMatrix>> rmsStopLayers;
    private final IGradientDescentProcessProvider<L> processProvider;
    private final Double rmsStopCoeff;
    private final Double epsilon;

    public GradientDescentWithRmsStopProcessProvider(IGradientDescentProcessProvider<L> processProvider, Double rmsStopCoeff) {
        this.rmsStopCoeff = rmsStopCoeff;
        this.epsilon = Math.pow(10, -8);
        this.processProvider = processProvider;
        rmsStopLayers = new ArrayList<>();
    }

    @Override
    public Function<GradientDescentCorrectionsContainer<L>, GradientDescentCorrectionsContainer<L>> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel<L> correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<L> layers = correctedNeuralNetworkModel.getLayers();
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initRmsStopLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parameterMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> rmsStopMatrices = rmsStopLayers.get(layerIndex);

                for (int parameterIndex = 0; parameterIndex < parameterMatrices.size(); parameterIndex++) {
                    //Sdw = c * Sdw + (1 - c)*dW²
                    rmsStopMatrices.set(parameterIndex, rmsStopMatrices.get(parameterIndex).muli(rmsStopCoeff).addi(MatrixFunctions.pow(gradientDescentCorrection.getCorrectionResults().get(parameterIndex), 2d).muli(1d - rmsStopCoeff)));
                    //W = W - a * dW/(sqrt(Sdw) + e)
                    DoubleMatrix weightCorrection = gradientDescentCorrection.getCorrectionResults().get(parameterIndex).div(MatrixFunctions.sqrt(rmsStopMatrices.get(parameterIndex)).addi(epsilon)).muli(container.getLearningRate());
                    parameterMatrices.get(parameterIndex).subi(weightCorrection);

                }
            }
            return new GradientDescentCorrectionsContainer<>(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    private List<List<DoubleMatrix>> initRmsStopLayers(List<L> layers) {
        return layers.stream()
                .map(layer -> layer.getParametersMatrix()
                        .stream()
                        .map(p -> DoubleMatrix.zeros(p.getRows(), p.getColumns()))
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }


    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return processProvider.getBackwardComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return processProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return processProvider.getFirstErrorComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer<L>, GradientLayerProvider<L>> getForwardComputationLauncher() {
        return processProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }
}
