package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class DropOutUtils {

    private DropOutUtils() {
    }

    public static DropOutUtils init() {
        return new DropOutUtils();
    }

    public List<DoubleMatrix> getDropOutMatrix(double[] dropOutParameters, List<WeightBiasLayer> layers) {
        List<DoubleMatrix> dropOutMatrixList = new ArrayList<>();

        WeightBiasLayer firstLayer = layers.get(0);
        double firstDropOutParameter1 = dropOutParameters[0];
        DoubleMatrix firstDropOutMatrix = DoubleMatrix.rand(1, firstLayer.getInputLayerSize()).lei(firstDropOutParameter1).divi(firstDropOutParameter1);
        dropOutMatrixList.add(firstDropOutMatrix);

        for (int layerIndex = 0, dropOutParameterIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutParameterIndex++) {
            WeightBiasLayer layer = layers.get(layerIndex);
            double dropOutParameter = dropOutParameters[dropOutParameterIndex];
            DoubleMatrix dropOutMatrix = DoubleMatrix.rand(1, layer.getOutputLayerSize()).lei(dropOutParameter).divi(dropOutParameter);
            dropOutMatrixList.add(dropOutMatrix);
        }
        return dropOutMatrixList;
    }
}
