package com.fgiannesini.neuralnetwork.activationfunctions;

public enum ActivationFunctionType {

    NONE {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new IdentityFunctionApplier();
        }
    },
    SIGMOID {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new SigmoidFunctionApplier();
        }
    },
    TANH {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new TanhFunctionApplier();
        }
    },
    RELU {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new ReluFunctionApplier();
        }
    },
    LEAKY_RELU {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new LeakyReluFunctionApplier();
        }
    },
    SOFT_MAX {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new SoftMaxFunctionApplier();
        }
    };

    public abstract ActivationFunctionApplier getActivationFunction();
}
