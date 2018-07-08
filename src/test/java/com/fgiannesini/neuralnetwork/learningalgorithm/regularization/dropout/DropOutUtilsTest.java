package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

class DropOutUtilsTest {

    @Test
    void getDropOutMatrix() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(3)
                .addLayer(2000)
                .addLayer(1000)
                .build();
        double[] dropOutParameters = {1, 0.4, 0.8};
        List<DoubleMatrix> dropOutMatrixList = DropOutUtils.init()
                .getDropOutMatrix(dropOutParameters, neuralNetworkModel.getLayers());

        Assertions.assertEquals(3, dropOutMatrixList.size());

        assertDropOutMatrix(dropOutMatrixList.get(0), 3, 1, 3);
        assertDropOutMatrix(dropOutMatrixList.get(1), 2000, 2.5, 800);
        assertDropOutMatrix(dropOutMatrixList.get(2), 1000, 1.25, 800);
    }

    private void assertDropOutMatrix(DoubleMatrix inputLayer, int expectedColumns, double expectedDropOutValue, int expectedPositiveCount) {
        Assertions.assertEquals(1, inputLayer.getRows());
        Assertions.assertEquals(expectedColumns, inputLayer.getColumns());
        long count = Arrays.stream(inputLayer.data).filter(d -> d > 0).count();
        Assertions.assertTrue(expectedPositiveCount > count * 0.9);
        Assertions.assertTrue(expectedPositiveCount < count * 1.1);
        Assertions.assertTrue(Arrays.stream(inputLayer.data).filter(d -> d > 0).allMatch(d -> d == expectedDropOutValue));
    }
}