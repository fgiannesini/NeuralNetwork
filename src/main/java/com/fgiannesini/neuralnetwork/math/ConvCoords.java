package com.fgiannesini.neuralnetwork.math;

public class ConvCoords {

    private final int rowIndex;
    private final int columnIndex;

    public ConvCoords(int rowIndex, int columnIndex) {
        this.rowIndex = rowIndex;
        this.columnIndex = columnIndex;
    }

    public int getRowIndex() {
        return rowIndex;
    }

    public int getColumnIndex() {
        return columnIndex;
    }
}
