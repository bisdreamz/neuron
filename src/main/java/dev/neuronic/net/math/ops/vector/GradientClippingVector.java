package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.GradientClipping;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vector implementation of GradientClipping.
 * This class is only loaded when Vector API is available.
 */
public final class GradientClippingVector implements GradientClipping.Impl {
    
    @Override
    public void clipByValue(float[] array, float maxValue) {
        if (!Vectorization.shouldVectorize(array.length)) {
            scalarClipByValue(array, maxValue);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int loopBound = Vectorization.loopBound(array.length);
        FloatVector maxVec = FloatVector.broadcast(species, maxValue);
        FloatVector minVec = FloatVector.broadcast(species, -maxValue);
        
        int i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vec = FloatVector.fromArray(species, array, i);
            vec = vec.max(minVec).min(maxVec);
            vec.intoArray(array, i);
        }
        
        // Handle remaining elements
        for (; i < array.length; i++) {
            array[i] = Math.max(-maxValue, Math.min(maxValue, array[i]));
        }
    }
    
    @Override
    public void scaleInPlace(float[] array, float scale) {
        if (!Vectorization.shouldVectorize(array.length)) {
            scalarScaleInPlace(array, scale);
            return;
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        int loopBound = Vectorization.loopBound(array.length);
        FloatVector scaleVec = FloatVector.broadcast(species, scale);
        
        int i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vec = FloatVector.fromArray(species, array, i);
            vec = vec.mul(scaleVec);
            vec.intoArray(array, i);
        }
        
        // Handle remaining elements
        for (; i < array.length; i++) {
            array[i] *= scale;
        }
    }
    
    private void scalarClipByValue(float[] array, float maxValue) {
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.max(-maxValue, Math.min(maxValue, array[i]));
        }
    }
    
    private void scalarScaleInPlace(float[] array, float scale) {
        for (int i = 0; i < array.length; i++) {
            array[i] *= scale;
        }
    }
}