package dev.neuronic.net.optimizers.adamw.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.optimizers.adamw.FusedAdamWUpdate;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized implementation of FusedAdamWUpdate.
 * This class is only loaded when Vector API is available.
 */
public final class FusedAdamWUpdateVector implements FusedAdamWUpdate.Impl {
    
    @Override
    public void compute(float[] params, float[] gradients, float[] momentum, float[] velocity,
                       float beta1, float beta2, float learningRate, float epsilon,
                       float weightDecay, float momentumCorrection, float velocityCorrection,
                       boolean applyWeightDecay) {
        
        if (Vectorization.shouldVectorize(params.length)) {
            computeVectorized(params, gradients, momentum, velocity, beta1, beta2, learningRate,
                            epsilon, weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
        } else {
            computeScalar(params, gradients, momentum, velocity, beta1, beta2, learningRate,
                        epsilon, weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
        }
    }
    
    private void computeVectorized(float[] params, float[] gradients, float[] momentum, float[] velocity,
                                 float beta1, float beta2, float learningRate, float epsilon,
                                 float weightDecay, float momentumCorrection, float velocityCorrection,
                                 boolean applyWeightDecay) {
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int len = params.length;
        int upper = Vectorization.loopBound(len);
        
        // Broadcast constants
        FloatVector beta1V = FloatVector.broadcast(species, beta1);
        FloatVector beta2V = FloatVector.broadcast(species, beta2);
        FloatVector oneMinusBeta1V = FloatVector.broadcast(species, 1.0f - beta1);
        FloatVector oneMinusBeta2V = FloatVector.broadcast(species, 1.0f - beta2);
        FloatVector learningRateV = FloatVector.broadcast(species, learningRate);
        FloatVector epsilonV = FloatVector.broadcast(species, epsilon);
        FloatVector momentumCorrectionV = FloatVector.broadcast(species, momentumCorrection);
        FloatVector velocityCorrectionV = FloatVector.broadcast(species, velocityCorrection);
        FloatVector oneMinusWeightDecayV = FloatVector.broadcast(species, 1.0f - weightDecay);
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            // Load vectors
            FloatVector paramV = FloatVector.fromArray(species, params, i);
            FloatVector gradV = FloatVector.fromArray(species, gradients, i);
            FloatVector momV = FloatVector.fromArray(species, momentum, i);
            FloatVector velV = FloatVector.fromArray(species, velocity, i);
            
            // Update momentum: m = β₁ * m + (1 - β₁) * g
            momV = momV.mul(beta1V).add(gradV.mul(oneMinusBeta1V));
            momV.intoArray(momentum, i);
            
            // Update velocity: v = β₂ * v + (1 - β₂) * g²
            FloatVector gradSquaredV = gradV.mul(gradV);
            velV = velV.mul(beta2V).add(gradSquaredV.mul(oneMinusBeta2V));
            velV.intoArray(velocity, i);
            
            // Bias correction
            FloatVector mHatV = momV.div(momentumCorrectionV);
            FloatVector vHatV = velV.div(velocityCorrectionV);
            
            // Parameter update: p = p - α * m̂ / (√v̂ + ε)
            FloatVector sqrtVHatV = vHatV.sqrt().add(epsilonV);
            paramV = paramV.sub(learningRateV.mul(mHatV).div(sqrtVHatV));
            
            // Weight decay: p = p * (1 - λ)
            if (applyWeightDecay) {
                paramV = paramV.mul(oneMinusWeightDecayV);
            }
            
            paramV.intoArray(params, i);
        }
        
        // Handle remaining elements
        for (; i < len; i++) {
            float grad = gradients[i];
            
            // Update momentum
            momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * grad;
            
            // Update velocity
            velocity[i] = beta2 * velocity[i] + (1.0f - beta2) * grad * grad;
            
            // Bias correction
            float mHat = momentum[i] / momentumCorrection;
            float vHat = velocity[i] / velocityCorrection;
            
            // Parameter update
            params[i] -= learningRate * mHat / (float)(Math.sqrt(vHat) + epsilon);
            
            // Weight decay
            if (applyWeightDecay) {
                params[i] *= (1.0f - weightDecay);
            }
        }
    }
    
    private void computeScalar(float[] params, float[] gradients, float[] momentum, float[] velocity,
                             float beta1, float beta2, float learningRate, float epsilon,
                             float weightDecay, float momentumCorrection, float velocityCorrection,
                             boolean applyWeightDecay) {
        
        float oneMinusBeta1 = 1.0f - beta1;
        float oneMinusBeta2 = 1.0f - beta2;
        float oneMinusWeightDecay = 1.0f - weightDecay;
        
        for (int i = 0; i < params.length; i++) {
            float grad = gradients[i];
            
            // Update momentum
            momentum[i] = beta1 * momentum[i] + oneMinusBeta1 * grad;
            
            // Update velocity
            velocity[i] = beta2 * velocity[i] + oneMinusBeta2 * grad * grad;
            
            // Bias correction
            float mHat = momentum[i] / momentumCorrection;
            float vHat = velocity[i] / velocityCorrection;
            
            // Parameter update
            params[i] -= learningRate * mHat / (float)(Math.sqrt(vHat) + epsilon);
            
            // Weight decay
            if (applyWeightDecay) {
                params[i] *= oneMinusWeightDecay;
            }
        }
    }
}