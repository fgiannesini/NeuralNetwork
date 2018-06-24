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
    },
    RELU {
        @Override
        public ActivationFunctionApplyer getActivationFunction() {
            return new ReluFunctionApplyer();
        }
    },
    LEAKY_RELU {
        @Override
        public ActivationFunctionApplyer getActivationFunction() {
            return new LeakyReluFunctionApplyer();
        }
    };

    public abstract ActivationFunctionApplyer getActivationFunction();
}
