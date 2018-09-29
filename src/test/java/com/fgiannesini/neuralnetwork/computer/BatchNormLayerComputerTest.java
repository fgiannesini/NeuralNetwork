package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

class BatchNormLayerComputerTest {

    @Test
    void compute_on_weight_matrix_and_batch_norm() {
        BatchNormLayer layer = new BatchNormLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        BatchNormData input = new BatchNormData(DoubleMatrix.ones(5).mul(3f), new MeanDeviationProvider());
        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        BatchNormData result = (BatchNormData) layerComputerVisitor.getIntermediateOutputResult().getResult();
        DoubleMatrixAssertions.assertMatrices(result.getInput(), new DoubleMatrix(2, 1, 1, 1));
    }

    @Test
    void compute_on_weight_matrix_and_batch_norm_with_three_inputs() {
        BatchNormLayer layer = new BatchNormLayer(4, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        BatchNormData input = new BatchNormData(new DoubleMatrix(4, 3, 1, 2, 3, 4, 5, 6, 7, 9, 9, 10, 11, 12), new MeanDeviationProvider());
        DoubleMatrix expectedOutput = new DoubleMatrix(2, 3, -0.24944, -0.24944, 1.05099, 1.05099, 2.19844, 2.19844);

        LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(input);
        layer.accept(layerComputerVisitor);
        BatchNormData result = (BatchNormData) layerComputerVisitor.getIntermediateOutputResult().getResult();
        DoubleMatrixAssertions.assertMatrices(result.getInput(), expectedOutput);

    }

}