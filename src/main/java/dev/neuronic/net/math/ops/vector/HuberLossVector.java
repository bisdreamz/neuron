package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.HuberLoss;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of HuberLoss.
 * This class is only loaded when Vector API is available.
 */
public final class HuberLossVector implements HuberLoss.Impl {
    
    @Override
    public float computeLoss(float[] predictions, float[] targets, float delta) {
        if (!Vectorization.shouldVectorize(predictions.length)) {
            return scalarComputeLoss(predictions, targets, delta);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector deltaVec = FloatVector.broadcast(species, delta);
        FloatVector halfDeltaSq = FloatVector.broadcast(species, 0.5f * delta * delta);
        FloatVector half = FloatVector.broadcast(species, 0.5f);
        
        int length = predictions.length;
        int L = species.length();
        int upper = Vectorization.loopBound(length);
        
        FloatVector sumVec = FloatVector.zero(species);
        
        for (int i = 0; i < upper; i += L) {
            FloatVector pred = FloatVector.fromArray(species, predictions, i);
            FloatVector targ = FloatVector.fromArray(species, targets, i);
            FloatVector error = pred.sub(targ);
            FloatVector absError = error.abs();
            
            VectorMask<Float> smallError = absError.compare(VectorOperators.LE, deltaVec);
            
            FloatVector quadLoss = error.mul(error).mul(half);
            FloatVector linLoss = deltaVec.mul(absError).sub(halfDeltaSq);
            
            FloatVector loss = linLoss.blend(quadLoss, smallError);
            
            sumVec = sumVec.add(loss);
        }
        
        float sum = sumVec.reduceLanes(VectorOperators.ADD);
        
        for (int i = upper; i < length; i++) {
            float error = predictions[i] - targets[i];
            float absError = Math.abs(error);
            
            if (absError <= delta)
                sum += 0.5f * error * error;
            else
                sum += delta * absError - 0.5f * delta * delta;
        }
        
        return sum / length;
    }
    
    @Override
    public void computeDerivatives(float[] predictions, float[] targets, float delta, float[] output) {
        if (!Vectorization.shouldVectorize(predictions.length)) {
            scalarComputeDerivatives(predictions, targets, delta, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector deltaVec = FloatVector.broadcast(species, delta);
        FloatVector negDelta = FloatVector.broadcast(species, -delta);
        
        int length = predictions.length;
        int L = species.length();
        int upper = Vectorization.loopBound(length);
        
        for (int i = 0; i < upper; i += L) {
            FloatVector pred = FloatVector.fromArray(species, predictions, i);
            FloatVector targ = FloatVector.fromArray(species, targets, i);
            FloatVector error = pred.sub(targ);
            
            VectorMask<Float> smallError = error.abs().compare(VectorOperators.LE, deltaVec);
            VectorMask<Float> positiveError = error.compare(VectorOperators.GT, 0);
            
            FloatVector derivative = negDelta.blend(deltaVec, positiveError);
            derivative = derivative.blend(error, smallError);
            
            derivative.intoArray(output, i);
        }
        
        for (int i = upper; i < length; i++) {
            float error = predictions[i] - targets[i];
            
            if (Math.abs(error) <= delta)
                output[i] = error;
            else
                output[i] = error > 0 ? delta : -delta;
        }
    }
    
    private float scalarComputeLoss(float[] predictions, float[] targets, float delta) {
        float sum = 0.0f;
        float halfDeltaSq = 0.5f * delta * delta;
        
        for (int i = 0; i < predictions.length; i++) {
            float error = predictions[i] - targets[i];
            float absError = Math.abs(error);
            
            if (absError <= delta)
                sum += 0.5f * error * error;
            else
                sum += delta * absError - halfDeltaSq;
        }
        
        return sum / predictions.length;
    }
    
    private void scalarComputeDerivatives(float[] predictions, float[] targets, float delta, float[] output) {
        for (int i = 0; i < predictions.length; i++) {
            float error = predictions[i] - targets[i];
            
            if (Math.abs(error) <= delta)
                output[i] = error;
            else
                output[i] = error > 0 ? delta : -delta;
        }
    }
}