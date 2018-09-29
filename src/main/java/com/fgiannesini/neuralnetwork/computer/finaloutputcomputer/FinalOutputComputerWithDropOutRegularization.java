package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization implements IFinalOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final List<Layer> layers;

    public FinalOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, List<Layer> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layers = layers;
    }

    public LayerTypeData compute(LayerTypeData input) {
        DataFunctionApplier dataFunctionApplier = new DataFunctionApplier(matrix -> matrix.dup().muliColumnVector(dropOutMatrixList.get(0)));
        input.accept(dataFunctionApplier);
        LayerTypeData regularizedInput = dataFunctionApplier.getLayerTypeData();
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(regularizedInput);
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            Layer layer = layers.get(layerIndex);
            LayerComputerWithDropOutRegularizationVisitor layerVisitor = new LayerComputerWithDropOutRegularizationVisitor(dropOutMatrixList.get(dropOutIndex), intermediateOutputResult.getResult());
            layer.accept(layerVisitor);
            intermediateOutputResult = layerVisitor.getIntermediateOutputResult();
        }
        return intermediateOutputResult.getResult();
    }

}
