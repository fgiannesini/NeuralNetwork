package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputer<L extends Layer> implements IFinalOutputComputer {

    private final ILayerComputer<L> layerComputer;
    private List<L> layers;

    public FinalOutputComputer(List<L> layers, ILayerComputer<L> layerComputer) {
        this.layers = layers;
        this.layerComputer = layerComputer;
    }

    @Override
    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix firstCurrentMatrix = inputMatrix.dup();
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstCurrentMatrix);
        for (L layer : layers) {
            intermediateOutputResult = layerComputer.computeAFromInput(intermediateOutputResult.getResult(), layer);
        }
        return intermediateOutputResult.getResult();
    }

}
