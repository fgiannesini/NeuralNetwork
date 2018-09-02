package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
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
        DoubleMatrix currentMatrix = inputMatrix.dup();
        for (L layer : layers) {
            currentMatrix = layerComputer.computeAFromInput(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
