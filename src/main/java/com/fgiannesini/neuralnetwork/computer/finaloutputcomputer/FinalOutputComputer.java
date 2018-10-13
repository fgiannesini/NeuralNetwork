package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputer implements IFinalOutputComputer {

    private final List<Layer> layers;

    public FinalOutputComputer(List<Layer> layers) {
        this.layers = layers;
    }

    @Override
    public LayerTypeData compute(LayerTypeData input) {
        DataFunctionApplier dataVisitor = new DataFunctionApplier(DoubleMatrix::dup);
        input.accept(dataVisitor);
        LayerTypeData firstData = dataVisitor.getLayerTypeData();

        ForwardDataAdapterVisitor firstDataAdaptorVisitor = new ForwardDataAdapterVisitor(firstData);
        layers.get(0).accept(firstDataAdaptorVisitor);
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstDataAdaptorVisitor.getData());

        for (Layer layer : layers) {
            LayerTypeData previousResult = intermediateOutputResult.getResult();
            ForwardDataAdapterVisitor dataAdaptorVisitor = new ForwardDataAdapterVisitor(previousResult);
            layer.accept(dataAdaptorVisitor);
            LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(dataAdaptorVisitor.getData());
            layer.accept(layerComputerVisitor);
            intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
        }
        return intermediateOutputResult.getResult();
    }

}
