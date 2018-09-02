package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization<L extends Layer> implements IFinalOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer<L> layerComputer;
    private List<L> layers;

    public FinalOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, ILayerComputer<L> layerComputer, List<L> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layerComputer = layerComputer;
        this.layers = layers;
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            L layer = layers.get(layerIndex);
            currentMatrix = layerComputer.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = layerComputer.computeAFromZ(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
