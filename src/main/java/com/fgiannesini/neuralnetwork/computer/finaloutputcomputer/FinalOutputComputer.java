package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.LayerComputerBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class FinalOutputComputer implements IFinalOutputComputer {

    private final NeuralNetworkModel<? extends Layer> model;
    private final ILayerComputer layerComputer;

    public FinalOutputComputer(NeuralNetworkModel<? extends Layer> model) {
        this.model = model;
        layerComputer = LayerComputerBuilder.init()
                .withLayerType(model.getLayerType())
                .build();
    }

    @Override
    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup();
        for (Layer layer : model.getLayers()) {
            currentMatrix = layerComputer.computeAFromInput(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
