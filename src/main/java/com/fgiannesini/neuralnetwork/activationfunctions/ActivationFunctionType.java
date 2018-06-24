package com.fgiannesini.neuralnetwork.activationfunctions;

public enum ActivationFunctionType {

    NONE {
        @Override
        public ActivationFunctionApplyer getActivationFunction() {
            return matrix -> matrix;
        }
    },
    SIGMOID {
        @Override
        public ActivationFunctionApplyer getActivationFunction() {
            return new SigmoidFunctionApplyer();
        }
    },
    TANH {
        @Override
        public ActivationFunctionApplyer getActivationFunction() {
            return new TanhFunctionApplyer();
        }
    };

    public abstract ActivationFunctionApplyer getActivationFunction();
}
