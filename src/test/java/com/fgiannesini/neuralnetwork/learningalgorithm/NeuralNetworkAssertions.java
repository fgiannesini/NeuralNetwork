package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.function.Executable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkAssertions {

    public static void checkSameNeuralNetworks(NeuralNetworkModel firstNeuralNetworkModel, NeuralNetworkModel secondNeuralNetworkModel) {
        Assertions.assertAll(
                () -> Assertions.assertEquals(firstNeuralNetworkModel.getOutputSize(), secondNeuralNetworkModel.getOutputSize()),
                () -> Assertions.assertEquals(firstNeuralNetworkModel.getInputSize(), secondNeuralNetworkModel.getInputSize())
        );

        List<Layer> firstLayers = firstNeuralNetworkModel.getLayers();
        List<Layer> secondLayers = secondNeuralNetworkModel.getLayers();
        List<Executable> executables = new ArrayList<>();
        for (int i = 1; i < secondLayers.size(); i++) {
            Layer firstLayer = firstLayers.get(i);
            Layer secondLayer = secondLayers.get(i);
            executables.addAll(getMatrixAssertions(firstLayer.getWeightMatrix(), secondLayer.getWeightMatrix(), "Weight matrix"));
            executables.addAll(getMatrixAssertions(firstLayer.getBiasMatrix(), secondLayer.getBiasMatrix(), "Bias matrix"));
        }
        Assertions.assertAll(executables);
    }

    private static List<Executable> getMatrixAssertions(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix, String assertedMatrix) {
        return Arrays.asList(
                () -> Assertions.assertEquals(expectedMatrix.getColumns(), currentMatrix.getColumns()),
                () -> Assertions.assertEquals(expectedMatrix.getRows(), currentMatrix.getRows()),
                () -> Assertions.assertArrayEquals(expectedMatrix.data, currentMatrix.data, 0.00001, assertedMatrix + " are differents")
        );
    }

    public static void checkNeuralNetworksLayer(NeuralNetworkModel neuralNetworkModel, int layerIndex, double[][] expectedWeights, double[] expectedBias) {
        DoubleMatrix expectedWeightsMatrix = new DoubleMatrix(expectedWeights).transpose();
        DoubleMatrix expectedBiasMatrix = new DoubleMatrix(expectedBias);
        Layer layer = neuralNetworkModel.getLayers().get(layerIndex);
        List<Executable> matrixAssertions = new ArrayList<>();
        matrixAssertions.addAll(getMatrixAssertions(layer.getWeightMatrix(), expectedWeightsMatrix, "WeightMatrix"));
        matrixAssertions.addAll(getMatrixAssertions(layer.getBiasMatrix(), expectedBiasMatrix, "Bias Matrix"));
        Assertions.assertAll(matrixAssertions);
    }

}
