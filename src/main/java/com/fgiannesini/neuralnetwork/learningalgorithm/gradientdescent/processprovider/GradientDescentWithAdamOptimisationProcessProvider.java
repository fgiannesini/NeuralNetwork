package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrectionsContainer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class GradientDescentWithAdamOptimisationProcessProvider implements IGradientDescentProcessProvider {
    private final Double momentumCoeff;
    private final Double rmsStopCoeff;
    private final IGradientDescentProcessProvider processProvider;
    private final List<List<DoubleMatrix>> momentumLayers;
    private final List<List<DoubleMatrix>> rmsStopLayers;
    private final Double epsilon;

    public GradientDescentWithAdamOptimisationProcessProvider(IGradientDescentProcessProvider processProvider, Double momentumCoeff, Double rmsStopCoeff) {
        this.momentumCoeff = momentumCoeff;
        this.rmsStopCoeff = rmsStopCoeff;
        this.processProvider = processProvider;
        momentumLayers = new ArrayList<>();
        rmsStopLayers = new ArrayList<>();
        this.epsilon = Math.pow(10, -8);
    }

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        return processProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            if (momentumLayers.isEmpty()) {
                momentumLayers.addAll(initLayers(layers));
            }
            if (rmsStopLayers.isEmpty()) {
                rmsStopLayers.addAll(initLayers(layers));
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> layerMatrices = layers.get(layerIndex).getParametersMatrix();
                List<DoubleMatrix> momentumMatrices = momentumLayers.get(layerIndex);
                List<DoubleMatrix> rmsStopMatrices = rmsStopLayers.get(layerIndex);

                for (int matrixIndex = 0; matrixIndex < layerMatrices.size(); matrixIndex++) {
                    DoubleMatrix matrixCorrections = gradientDescentCorrection.getCorrectionResults().get(matrixIndex);

                    //Vdw = m*Vdw + (1 - m)*dW
                    momentumMatrices.set(matrixIndex, momentumMatrices.get(matrixIndex).muli(momentumCoeff).addi(matrixCorrections.mul(1d - momentumCoeff)));
                    //Sdw = c * Sdw + (1 - c)*dW²
                    rmsStopMatrices.set(matrixIndex, rmsStopMatrices.get(matrixIndex).muli(rmsStopCoeff).addi(MatrixFunctions.pow(matrixCorrections, 2d).muli(1d - rmsStopCoeff)));
                    //W = W - a * Vdw/(1-m)/(sqrt(Sdw/(1-c)) + e)
                    DoubleMatrix parameterCorrection = momentumMatrices.get(matrixIndex).div(1d - momentumCoeff).divi(MatrixFunctions.sqrt(rmsStopMatrices.get(matrixIndex).div(1d - rmsStopCoeff)).addi(epsilon)).muli(container.getLearningRate());
                    layerMatrices.get(matrixIndex).subi(parameterCorrection);
                }
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
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

}
