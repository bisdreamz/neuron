package dev.neuronic.net.math.ops.vector;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.ops.VectorClip;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

/**
 * Vector implementation of VectorClip.
 * This class is only loaded when Vector API is available.
 */
public final class VectorClipVector implements VectorClip.Impl {
    
    @Override
    public boolean clipInPlace(float[] array, float minValue, float maxValue) {
        if (!Vectorization.shouldVectorize(array.length)) {
            return scalarClipInPlace(array, minValue, maxValue);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector minVec = FloatVector.broadcast(species, minValue);
        FloatVector maxVec = FloatVector.broadcast(species, maxValue);
        
        boolean clipped = false;
        int i = 0;
        int upperBound = Vectorization.loopBound(array.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector values = FloatVector.fromArray(species, array, i);
            FloatVector original = values;
            
            values = values.max(minVec).min(maxVec);
            
            if (!clipped && !values.compare(VectorOperators.EQ, original).allTrue()) {
                clipped = true;
            }
            
            values.intoArray(array, i);
        }
        
        for (; i < array.length; i++) {
            float original = array[i];
            if (original < minValue) {
                array[i] = minValue;
                clipped = true;
            } else if (original > maxValue) {
                array[i] = maxValue;
                clipped = true;
            }
        }
        
        return clipped;
    }
    
    @Override
    public boolean wouldClip(float[] array, float minValue, float maxValue) {
        if (!Vectorization.shouldVectorize(array.length)) {
            return scalarWouldClip(array, minValue, maxValue);
        }
        
        VectorSpecies<Float> species = Vectorization.getSpecies();
        FloatVector minVec = FloatVector.broadcast(species, minValue);
        FloatVector maxVec = FloatVector.broadcast(species, maxValue);
        
        int i = 0;
        int upperBound = Vectorization.loopBound(array.length);
        
        for (; i < upperBound; i += species.length()) {
            FloatVector values = FloatVector.fromArray(species, array, i);
            
            if (values.compare(VectorOperators.LT, minVec).anyTrue() || 
                values.compare(VectorOperators.GT, maxVec).anyTrue()) {
                return true;
            }
        }
        
        for (; i < array.length; i++) {
            float value = array[i];
            if (value < minValue || value > maxValue) {
                return true;
            }
        }
        
        return false;
    }
    
    private boolean scalarClipInPlace(float[] array, float minValue, float maxValue) {
        boolean clipped = false;
        
        for (int i = 0; i < array.length; i++) {
            float original = array[i];
            if (original < minValue) {
                array[i] = minValue;
                clipped = true;
            } else if (original > maxValue) {
                array[i] = maxValue;
                clipped = true;
            }
        }
        
        return clipped;
    }
    
    private boolean scalarWouldClip(float[] array, float minValue, float maxValue) {
        for (float value : array) {
            if (value < minValue || value > maxValue) {
                return true;
            }
        }
        return false;
    }
}