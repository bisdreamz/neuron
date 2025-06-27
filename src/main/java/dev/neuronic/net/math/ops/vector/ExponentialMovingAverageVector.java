package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.ExponentialMovingAverage;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of ExponentialMovingAverage.
 * This class is only loaded when Vector API is available.
 */
public final class ExponentialMovingAverageVector implements ExponentialMovingAverage.Impl {
    
    @Override
    public void computeInPlace(float[] current, float[] newValues, float decay) {
        if (!Vectorization.shouldVectorize(current.length)) {
            scalarComputeInPlace(current, newValues, decay);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector decayVec = FloatVector.broadcast(species, decay);
        FloatVector oneMinusDecayVec = FloatVector.broadcast(species, 1.0f - decay);
        int upperBound = Vectorization.loopBound(current.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector currentVec = FloatVector.fromArray(species, current, i);
            FloatVector newVec = FloatVector.fromArray(species, newValues, i);
            
            newVec.fma(oneMinusDecayVec, currentVec.mul(decayVec)).intoArray(current, i);
        }
        
        float oneMinusDecay = 1.0f - decay;
        for (; i < current.length; i++) {
            current[i] = decay * current[i] + oneMinusDecay * newValues[i];
        }
    }
    
    @Override
    public void compute(float[] current, float[] newValues, float decay, float[] output) {
        if (!Vectorization.shouldVectorize(current.length)) {
            scalarCompute(current, newValues, decay, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector decayVec = FloatVector.broadcast(species, decay);
        FloatVector oneMinusDecayVec = FloatVector.broadcast(species, 1.0f - decay);
        int upperBound = Vectorization.loopBound(current.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector currentVec = FloatVector.fromArray(species, current, i);
            FloatVector newVec = FloatVector.fromArray(species, newValues, i);
            
            newVec.fma(oneMinusDecayVec, currentVec.mul(decayVec)).intoArray(output, i);
        }
        
        float oneMinusDecay = 1.0f - decay;
        for (; i < current.length; i++) {
            output[i] = decay * current[i] + oneMinusDecay * newValues[i];
        }
    }
    
    private void scalarComputeInPlace(float[] current, float[] newValues, float decay) {
        float oneMinusDecay = 1.0f - decay;
        for (int i = 0; i < current.length; i++) {
            current[i] = decay * current[i] + oneMinusDecay * newValues[i];
        }
    }
    
    private void scalarCompute(float[] current, float[] newValues, float decay, float[] output) {
        float oneMinusDecay = 1.0f - decay;
        for (int i = 0; i < current.length; i++) {
            output[i] = decay * current[i] + oneMinusDecay * newValues[i];
        }
    }
}