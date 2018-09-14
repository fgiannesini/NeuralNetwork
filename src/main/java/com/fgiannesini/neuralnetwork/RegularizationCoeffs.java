package com.fgiannesini.neuralnetwork;

import java.util.Arrays;

public class RegularizationCoeffs {

    private Double l2RegularizationCoeff;

    private double[] dropOutRegularizationCoeffs;

    public Double getL2RegularizationCoeff() {
        return l2RegularizationCoeff;
    }

    public void setL2RegularizationCoeff(Double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
    }

    public double[] getDropOutRegularizationCoeffs() {
        return dropOutRegularizationCoeffs;
    }

    public void setDropOutRegularizationCoeffs(double[] dropOutRegularizationCoeffs) {
        this.dropOutRegularizationCoeffs = dropOutRegularizationCoeffs;
    }

    @Override
    public String toString() {
        return "RegularizationCoeffs{" +
                "l2RegularizationCoeff=" + l2RegularizationCoeff +
                ", dropOutRegularizationCoeffs=" + Arrays.toString(dropOutRegularizationCoeffs) +
                '}';
    }
}
