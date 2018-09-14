package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization<L extends Layer> implements IFinalOutputComputer<L> {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer<L> layerComputer;
    private List<L> layers;

    public FinalOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, ILayerComputer<L> layerComputer, List<L> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layerComputer = layerComputer;
        this.layers = layers;
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix firstCurrentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstCurrentMatrix);
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            L layer = layers.get(layerIndex);
            intermediateOutputResult = layerComputer.computeZFromInput(intermediateOutputResult.getResult(), layer);
            DoubleMatrix currentMatrix = intermediateOutputResult.getResult();
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = layerComputer.computeAFromZ(currentMatrix, layer);
            intermediateOutputResult.setResult(currentMatrix);
        }
        return intermediateOutputResult.getResult();
    }

}
