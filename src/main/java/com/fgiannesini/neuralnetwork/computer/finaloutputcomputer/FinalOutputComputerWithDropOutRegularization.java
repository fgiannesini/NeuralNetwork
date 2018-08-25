package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.LayerComputerBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization implements IFinalOutputComputer {

    private final NeuralNetworkModel<? extends Layer> model;
    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer layerComputer;

    public FinalOutputComputerWithDropOutRegularization(NeuralNetworkModel<? extends Layer> model, List<DoubleMatrix> dropOutMatrixList) {
        this.model = model;
        this.dropOutMatrixList = dropOutMatrixList;
        layerComputer = LayerComputerBuilder.init()
                .withLayerType(model.getLayerType())
                .build();
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < model.getLayers().size(); layerIndex++, dropOutIndex++) {
            Layer layer = model.getLayers().get(layerIndex);
            currentMatrix = layerComputer.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = layerComputer.computeAFromZ(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
