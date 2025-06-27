package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.FusedMultiplyDivideSubtract;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of FusedMultiplyDivideSubtract.
 * This class is only loaded when Vector API is available.
 */
public final class FusedMultiplyDivideSubtractVector implements FusedMultiplyDivideSubtract.Impl {
    
    @Override
    public void compute(float[] params, float[] numerator, float[] denominator, float scale) {
        if (!Vectorization.shouldVectorize(params.length)) {
            scalarCompute(params, numerator, denominator, scale);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleVec = FloatVector.broadcast(species, scale);
        int upperBound = Vectorization.loopBound(params.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector paramsVec = FloatVector.fromArray(species, params, i);
            FloatVector numVec = FloatVector.fromArray(species, numerator, i);
            FloatVector denVec = FloatVector.fromArray(species, denominator, i);
            
            FloatVector updateVec = numVec.div(denVec).mul(scaleVec);
            paramsVec.sub(updateVec).intoArray(params, i);
        }
        
        for (; i < params.length; i++) {
            params[i] -= scale * (numerator[i] / denominator[i]);
        }
    }
    
    @Override
    public void computeAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        if (!Vectorization.shouldVectorize(params.length)) {
            scalarComputeAdd(params, numerator, denominator, scale);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scaleVec = FloatVector.broadcast(species, scale);
        int upperBound = Vectorization.loopBound(params.length);
        
        int i = 0;
        for (; i < upperBound; i += species.length()) {
            FloatVector paramsVec = FloatVector.fromArray(species, params, i);
            FloatVector numVec = FloatVector.fromArray(species, numerator, i);
            FloatVector denVec = FloatVector.fromArray(species, denominator, i);
            
            FloatVector updateVec = numVec.div(denVec).mul(scaleVec);
            paramsVec.add(updateVec).intoArray(params, i);
        }
        
        for (; i < params.length; i++) {
            params[i] += scale * (numerator[i] / denominator[i]);
        }
    }
    
    private void scalarCompute(float[] params, float[] numerator, float[] denominator, float scale) {
        for (int i = 0; i < params.length; i++) {
            params[i] -= scale * (numerator[i] / denominator[i]);
        }
    }
    
    private void scalarComputeAdd(float[] params, float[] numerator, float[] denominator, float scale) {
        for (int i = 0; i < params.length; i++) {
            params[i] += scale * (numerator[i] / denominator[i]);
        }
    }
}