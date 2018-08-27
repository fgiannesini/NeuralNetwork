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

    public static void checkSameNeuralNetworks(NeuralNetworkModel<? extends Layer> firstNeuralNetworkModel, NeuralNetworkModel<? extends Layer> secondNeuralNetworkModel) {
        Assertions.assertAll(
                () -> Assertions.assertEquals(firstNeuralNetworkModel.getOutputSize(), secondNeuralNetworkModel.getOutputSize()),
                () -> Assertions.assertEquals(firstNeuralNetworkModel.getInputSize(), secondNeuralNetworkModel.getInputSize())
        );

        List<? extends Layer> firstLayers = firstNeuralNetworkModel.getLayers();
        List<? extends Layer> secondLayers = secondNeuralNetworkModel.getLayers();
        List<Executable> executables = new ArrayList<>();
        for (int i = 0; i < secondLayers.size(); i++) {
            List<DoubleMatrix> firstParameterMatrix = firstLayers.get(i).getParametersMatrix();
            List<DoubleMatrix> secondParameterMatrix = secondLayers.get(i).getParametersMatrix();
            Assertions.assertEquals(firstParameterMatrix.size(), secondParameterMatrix.size());
            for (int j = 0; j < secondParameterMatrix.size(); j++) {
                executables.addAll(getMatrixAssertions(firstParameterMatrix.get(j), secondParameterMatrix.get(j)));
            }
        }
        Assertions.assertAll(executables);
    }

    private static List<Executable> getMatrixAssertions(DoubleMatrix currentMatrix, DoubleMatrix expectedMatrix) {
        return Arrays.asList(
                () -> Assertions.assertEquals(expectedMatrix.getColumns(), currentMatrix.getColumns()),
                () -> Assertions.assertEquals(expectedMatrix.getRows(), currentMatrix.getRows()),
                () -> Assertions.assertArrayEquals(expectedMatrix.data, currentMatrix.data, 0.00001)
        );
    }

    public static void checkNeuralNetworksLayer(NeuralNetworkModel<? extends Layer> neuralNetworkModel, int layerIndex, List<DoubleMatrix> expectedParametersMatrix) {
        Layer layer = neuralNetworkModel.getLayers().get(layerIndex);
        List<Executable> matrixAssertions = new ArrayList<>();
        List<DoubleMatrix> parametersMatrix = layer.getParametersMatrix();
        for (int i = 0; i < parametersMatrix.size(); i++) {
            matrixAssertions.addAll(getMatrixAssertions(parametersMatrix.get(i), expectedParametersMatrix.get(i)));
        }
        Assertions.assertAll(matrixAssertions);
    }

}
