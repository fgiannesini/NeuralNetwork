package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization<L extends Layer> implements IIntermediateOutputComputer<L> {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer<L> layerComputer;
    private List<L> layers;

    public IntermediateOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, ILayerComputer<L> layerComputer, List<L> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layerComputer = layerComputer;
        this.layers = layers;
    }

    public List<IntermediateOutputResult> compute(DoubleMatrix inputMatrix) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();

        DoubleMatrix firstMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstMatrix);
        intermediateOutputResults.add(intermediateOutputResult);

        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            L layer = layers.get(layerIndex);
            intermediateOutputResult = layerComputer.computeZFromInput(intermediateOutputResult.getResult(), layer);
            DoubleMatrix currentResult = intermediateOutputResult.getResult();
            currentResult.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentResult = layerComputer.computeAFromZ(currentResult, layer);
            intermediateOutputResult.setResult(currentResult);
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
