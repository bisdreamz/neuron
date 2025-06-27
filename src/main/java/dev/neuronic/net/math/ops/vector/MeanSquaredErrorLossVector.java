package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.MeanSquaredErrorLoss;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of MeanSquaredErrorLoss.
 * This class is only loaded when Vector API is available.
 */
public final class MeanSquaredErrorLossVector implements MeanSquaredErrorLoss.Impl {
    
    @Override
    public float computeLoss(float[] predictions, float[] targets) {
        if (!Vectorization.shouldVectorize(predictions.length)) {
            return scalarComputeLoss(predictions, targets);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector sumVector = FloatVector.zero(species);
        int i = 0;
        int upperBound = Vectorization.loopBound(predictions.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector pred = FloatVector.fromArray(species, predictions, i);
            FloatVector target = FloatVector.fromArray(species, targets, i);
            FloatVector diff = pred.sub(target);
            sumVector = sumVector.add(diff.mul(diff));
        }
        
        float sum = sumVector.reduceLanes(VectorOperators.ADD);
        
        for (; i < predictions.length; i++) {
            float diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        
        return sum / predictions.length;
    }
    
    @Override
    public void computeDerivatives(float[] predictions, float[] targets, float[] output) {
        if (!Vectorization.shouldVectorize(predictions.length)) {
            scalarComputeDerivatives(predictions, targets, output);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector scale = FloatVector.broadcast(species, 2.0f / predictions.length);
        int i = 0;
        int upperBound = Vectorization.loopBound(predictions.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector pred = FloatVector.fromArray(species, predictions, i);
            FloatVector target = FloatVector.fromArray(species, targets, i);
            pred.sub(target).mul(scale).intoArray(output, i);
        }
        
        float scalarScale = 2.0f / predictions.length;
        for (; i < predictions.length; i++) {
            output[i] = scalarScale * (predictions[i] - targets[i]);
        }
    }
    
    private float scalarComputeLoss(float[] predictions, float[] targets) {
        float sum = 0.0f;
        for (int i = 0; i < predictions.length; i++) {
            float diff = predictions[i] - targets[i];
            sum += diff * diff;
        }
        return sum / predictions.length;
    }
    
    private void scalarComputeDerivatives(float[] predictions, float[] targets, float[] output) {
        float scale = 2.0f / predictions.length;
        for (int i = 0; i < predictions.length; i++) {
            output[i] = scale * (predictions[i] - targets[i]);
        }
    }
}