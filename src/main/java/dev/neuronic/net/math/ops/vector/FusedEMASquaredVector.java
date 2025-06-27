package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.FusedEMASquared;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized implementation of FusedEMASquared.
 * This class is only loaded when Vector API is available.
 */
public final class FusedEMASquaredVector implements FusedEMASquared.Impl {
    
    @Override
    public void compute(float[] state, float[] gradients, float beta) {
        if (Vectorization.shouldVectorize(state.length)) {
            computeVectorized(state, gradients, beta);
        } else {
            computeScalar(state, gradients, beta);
        }
    }
    
    private void computeVectorized(float[] state, float[] gradients, float beta) {
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int len = state.length;
        int upper = Vectorization.loopBound(len);
        
        FloatVector betaV = FloatVector.broadcast(species, beta);
        FloatVector oneMinusBetaV = FloatVector.broadcast(species, 1.0f - beta);
        
        int i = 0;
        for (; i < upper; i += species.length()) {
            FloatVector stateV = FloatVector.fromArray(species, state, i);
            FloatVector gradV = FloatVector.fromArray(species, gradients, i);
            
            // Compute gradient squared and update state in one operation
            FloatVector gradSquaredV = gradV.mul(gradV);
            FloatVector newStateV = stateV.mul(betaV).add(gradSquaredV.mul(oneMinusBetaV));
            
            newStateV.intoArray(state, i);
        }
        
        // Handle remaining elements
        float oneMinusBeta = 1.0f - beta;
        for (; i < len; i++) {
            float gradSquared = gradients[i] * gradients[i];
            state[i] = beta * state[i] + oneMinusBeta * gradSquared;
        }
    }
    
    private void computeScalar(float[] state, float[] gradients, float beta) {
        float oneMinusBeta = 1.0f - beta;
        
        for (int i = 0; i < state.length; i++) {
            float gradSquared = gradients[i] * gradients[i];
            state[i] = beta * state[i] + oneMinusBeta * gradSquared;
        }
    }
}