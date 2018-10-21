package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.InputDropOutRegularizationVisitor;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization implements IFinalOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final List<Layer> layers;

    public FinalOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, List<Layer> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layers = layers;
    }

    public LayerTypeData compute(LayerTypeData input) {
        InputDropOutRegularizationVisitor inputVisitor = new InputDropOutRegularizationVisitor(dropOutMatrixList.get(0));
        input.accept(inputVisitor);
        LayerTypeData regularizedInput = inputVisitor.getRegularizedData();
        ForwardDataAdapterVisitor firstDataAdaptorVisitor = new ForwardDataAdapterVisitor(regularizedInput);
        layers.get(0).accept(firstDataAdaptorVisitor);
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstDataAdaptorVisitor.getData());
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            Layer layer = layers.get(layerIndex);
            LayerTypeData previousResult = intermediateOutputResult.getResult();
            ForwardDataAdapterVisitor dataAdaptorVisitor = new ForwardDataAdapterVisitor(previousResult);
            layer.accept(dataAdaptorVisitor);
            LayerComputerWithDropOutRegularizationVisitor layerVisitor = new LayerComputerWithDropOutRegularizationVisitor(dropOutMatrixList.get(dropOutIndex), dataAdaptorVisitor.getData());
            layer.accept(layerVisitor);
            intermediateOutputResult = layerVisitor.getIntermediateOutputResult();
        }
        return intermediateOutputResult.getResult();
    }

}
