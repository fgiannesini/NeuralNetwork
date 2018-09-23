package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class DropOutUtils {

    private DropOutUtils() {
    }

    public static DropOutUtils init() {
        return new DropOutUtils();
    }

    public List<DoubleMatrix> getDropOutMatrix(double[] dropOutParameters, List<Layer> layers) {
        List<DoubleMatrix> dropOutMatrixList = new ArrayList<>();

        FirstDropOutVisitor firstDropOutVisitor = new FirstDropOutVisitor(dropOutParameters[0]);
        layers.get(0).accept(firstDropOutVisitor);
        dropOutMatrixList.add(firstDropOutVisitor.getFirstDropOutMatrix());

        for (int layerIndex = 0, dropOutParameterIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutParameterIndex++) {
            Layer layer = layers.get(layerIndex);
            double dropOutParameter = dropOutParameters[dropOutParameterIndex];
            DropOutVisitor dropOutVisitor = new DropOutVisitor(dropOutParameter);
            layer.accept(firstDropOutVisitor);
            dropOutMatrixList.add(dropOutVisitor.getDropOutMatrix());
        }
        return dropOutMatrixList;
    }
}
