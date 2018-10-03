package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import org.jblas.DoubleMatrix;

public class DataExtractorVisitor implements DataVisitor {

    private DoubleMatrix data;

    @Override
    public void visit(WeightBiasData input) {
        data = input.getData();
    }

    @Override
    public void visit(BatchNormData input) {
        data = input.getInput();
    }

    public DoubleMatrix getData() {
        return data;
    }
}
