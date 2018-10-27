package com.fgiannesini.neuralnetwork;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

public class RegularizationCoeffs implements Serializable {

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

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RegularizationCoeffs)) return false;
        RegularizationCoeffs that = (RegularizationCoeffs) o;
        return Objects.equals(l2RegularizationCoeff, that.l2RegularizationCoeff) &&
                Arrays.equals(dropOutRegularizationCoeffs, that.dropOutRegularizationCoeffs);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(l2RegularizationCoeff);
        result = 31 * result + Arrays.hashCode(dropOutRegularizationCoeffs);
        return result;
    }
}
