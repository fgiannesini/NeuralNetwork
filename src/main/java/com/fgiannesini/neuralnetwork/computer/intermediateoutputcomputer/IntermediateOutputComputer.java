package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputer<L extends Layer> implements IIntermediateOutputComputer<L> {

    private final NeuralNetworkModel<L> model;
    private final ILayerComputer<L> layerComputer;

    public IntermediateOutputComputer(NeuralNetworkModel<L> model, ILayerComputer<L> layerComputer) {
        this.model = model;
        this.layerComputer = layerComputer;
    }

    public List<IntermediateOutputResult> compute(DoubleMatrix inputMatrix) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(inputMatrix.dup());
        intermediateOutputResults.add(intermediateOutputResult);
        for (L layer : model.getLayers()) {
            intermediateOutputResult = layerComputer.computeAFromInput(intermediateOutputResult.getResult(), layer);
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
