package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.LayerComputerWithDropOutRegularizationVisitor;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization implements IIntermediateOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final List<Layer> layers;

    public IntermediateOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, List<Layer> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layers = layers;
    }

    public List<IntermediateOutputResult> compute(LayerTypeData data) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();

        DataFunctionApplier dataVisitor = new DataFunctionApplier(matrix -> matrix.dup().muliColumnVector(dropOutMatrixList.get(0)));
        data.accept(dataVisitor);
        LayerTypeData regularizedInput = dataVisitor.getLayerTypeData();
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(regularizedInput);
        intermediateOutputResults.add(intermediateOutputResult);

        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            LayerComputerWithDropOutRegularizationVisitor computerVisitor = new LayerComputerWithDropOutRegularizationVisitor(dropOutMatrixList.get(dropOutIndex), intermediateOutputResult.getResult());
            layers.get(layerIndex).accept(computerVisitor);
            intermediateOutputResult = computerVisitor.getIntermediateOutputResult();
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
