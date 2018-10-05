package com.fgiannesini.neuralnetwork.batch;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

public class BatchIteratorVisitor implements DataVisitor {
    private final int lowerIndex;
    private final int upperIndex;
    private LayerTypeData subData;

    public BatchIteratorVisitor(int lowerIndex, int upperIndex) {
        this.lowerIndex = lowerIndex;
        this.upperIndex = upperIndex;
    }

    @Override
    public void visit(WeightBiasData data) {
        DoubleMatrix subMatrix = data.getData().getColumns(new IntervalRange(lowerIndex, upperIndex));
        subData = new WeightBiasData(subMatrix);
    }

    @Override
    public void visit(BatchNormData data) {
        DoubleMatrix subMatrix = data.getData().getColumns(new IntervalRange(lowerIndex, upperIndex));
        subData = new BatchNormData(subMatrix, data.getMeanDeviationProvider());
    }

    public LayerTypeData getSubData() {
        return subData;
    }
}
