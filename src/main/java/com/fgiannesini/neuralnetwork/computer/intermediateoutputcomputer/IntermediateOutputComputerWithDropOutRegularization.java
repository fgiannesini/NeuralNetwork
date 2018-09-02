package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization<L extends Layer> implements IIntermediateOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer layerComputer;
    private List<L> layers;

    public IntermediateOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, ILayerComputer layerComputer, List<L> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layerComputer = layerComputer;
        this.layers = layers;
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        intermediateMatrix.add(currentMatrix);
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            L layer = layers.get(layerIndex);
            currentMatrix = layerComputer.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = layerComputer.computeAFromZ(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
