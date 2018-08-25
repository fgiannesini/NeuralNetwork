package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.LayerComputerBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputer implements IIntermediateOutputComputer {

    private final NeuralNetworkModel<? extends Layer> model;
    private final ILayerComputer layerComputer;

    public IntermediateOutputComputer(NeuralNetworkModel<? extends Layer> model) {
        this.model = model;
        layerComputer = LayerComputerBuilder.init()
                .withLayerType(model.getLayerType())
                .build();
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup();
        intermediateMatrix.add(currentMatrix);
        for (Layer layer : model.getLayers()) {
            currentMatrix = layerComputer.computeAFromInput(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
