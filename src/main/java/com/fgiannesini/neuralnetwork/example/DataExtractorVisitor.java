package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import org.jblas.DoubleMatrix;

public class DataExtractorVisitor implements DataVisitor {

    private DoubleMatrix data;

    @Override
    public void visit(WeightBiasData input) {
        data = input.getData();
    }

    @Override
    public void visit(BatchNormData input) {
        data = input.getData();
    }

    public DoubleMatrix getData() {
        return data;
    }
}
