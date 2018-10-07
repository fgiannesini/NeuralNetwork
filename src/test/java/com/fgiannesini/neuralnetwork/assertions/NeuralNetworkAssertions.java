package com.fgiannesini.neuralnetwork.assertions;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.function.Executable;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkAssertions {

    public static void checkSameNeuralNetworks(NeuralNetworkModel firstNeuralNetworkModel, NeuralNetworkModel secondNeuralNetworkModel) {
        List<Layer> firstLayers = firstNeuralNetworkModel.getLayers();
        List<Layer> secondLayers = secondNeuralNetworkModel.getLayers();
        List<Executable> executables = new ArrayList<>();
        for (int i = 0; i < secondLayers.size(); i++) {

            List<DoubleMatrix> firstParameterMatrix = firstLayers.get(i).getParametersMatrix();
            List<DoubleMatrix> secondParameterMatrix = secondLayers.get(i).getParametersMatrix();
            Assertions.assertEquals(firstParameterMatrix.size(), secondParameterMatrix.size());
            for (int j = 0; j < secondParameterMatrix.size(); j++) {
                DoubleMatrix currentMatrix = firstParameterMatrix.get(j);
                DoubleMatrix expectedMatrix = secondParameterMatrix.get(j);
                DoubleMatrixAssertions.assertMatrices(currentMatrix, expectedMatrix);
            }
        }
        Assertions.assertAll(executables);
    }

    public static void checkNeuralNetworksLayer(NeuralNetworkModel neuralNetworkModel, int layerIndex, List<DoubleMatrix> expectedParametersMatrix) {
        Layer layer = neuralNetworkModel.getLayers().get(layerIndex);
        List<DoubleMatrix> parametersMatrix = layer.getParametersMatrix();
        for (int i = 0; i < parametersMatrix.size(); i++) {
            DoubleMatrix currentMatrix = parametersMatrix.get(i);
            DoubleMatrix expectedMatrix = expectedParametersMatrix.get(i);
            DoubleMatrixAssertions.assertMatrices(currentMatrix, expectedMatrix);
        }
    }

}
