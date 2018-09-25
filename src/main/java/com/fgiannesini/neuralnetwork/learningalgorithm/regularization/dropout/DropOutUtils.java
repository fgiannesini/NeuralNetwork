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

        FirstDropOutMatrixGeneratorVisitor firstDropOutMatrixGeneratorVisitor = new FirstDropOutMatrixGeneratorVisitor(dropOutParameters[0]);
        layers.get(0).accept(firstDropOutMatrixGeneratorVisitor);
        dropOutMatrixList.add(firstDropOutMatrixGeneratorVisitor.getFirstDropOutMatrix());

        for (int layerIndex = 0, dropOutParameterIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutParameterIndex++) {
            Layer layer = layers.get(layerIndex);
            double dropOutParameter = dropOutParameters[dropOutParameterIndex];
            DropOutMatrixGeneratorVisitor dropOutMatrixGeneratorVisitor = new DropOutMatrixGeneratorVisitor(dropOutParameter);
            layer.accept(firstDropOutMatrixGeneratorVisitor);
            dropOutMatrixList.add(dropOutMatrixGeneratorVisitor.getDropOutMatrix());
        }
        return dropOutMatrixList;
    }
}
