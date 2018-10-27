package com.fgiannesini.neuralnetwork.computer.data.adapter;

import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

class DataAdapterComputerTest {

    @Test
    void convertMatrixToMatrixList() {
        DoubleMatrix input = new DoubleMatrix(12, 2, IntStream.range(0, 24).asDoubleStream().toArray());
        List<DoubleMatrix> doubleMatrices = DataAdapterComputer.get().convertMatrixToMatrixList(input, 2, 2, 3);

        Assertions.assertEquals(6, doubleMatrices.size());
        Assertions.assertEquals(doubleMatrices.get(0), new DoubleMatrix(2, 2, IntStream.range(0, 4).asDoubleStream().toArray()));
        Assertions.assertEquals(doubleMatrices.get(1), new DoubleMatrix(2, 2, IntStream.range(4, 8).asDoubleStream().toArray()));
        Assertions.assertEquals(doubleMatrices.get(2), new DoubleMatrix(2, 2, IntStream.range(8, 12).asDoubleStream().toArray()));
        Assertions.assertEquals(doubleMatrices.get(3), new DoubleMatrix(2, 2, IntStream.range(12, 16).asDoubleStream().toArray()));
        Assertions.assertEquals(doubleMatrices.get(4), new DoubleMatrix(2, 2, IntStream.range(16, 20).asDoubleStream().toArray()));
        Assertions.assertEquals(doubleMatrices.get(5), new DoubleMatrix(2, 2, IntStream.range(20, 24).asDoubleStream().toArray()));

        DoubleMatrix output = DataAdapterComputer.get().convertMatrixListToMatrix(doubleMatrices, 12);

        Assertions.assertEquals(input, output);
    }

    @Test
    void adaptMatrix() {
        List<DoubleMatrix> input = Arrays.asList(
                new DoubleMatrix(4, 4, IntStream.range(0, 16).asDoubleStream().toArray()),
                new DoubleMatrix(4, 4, IntStream.range(16, 32).asDoubleStream().toArray())
        );

        List<DoubleMatrix> bigger = DataAdapterComputer.get().adaptMatrices(input, 5, 5);
        List<DoubleMatrix> lower = DataAdapterComputer.get().adaptMatrices(bigger, 4, 4);
        DoubleMatrixAssertions.assertMatrices(input, lower);

        List<DoubleMatrix> output = DataAdapterComputer.get().adaptMatrices(bigger, 4, 4);

        DoubleMatrixAssertions.assertMatrices(input, output);
    }
}